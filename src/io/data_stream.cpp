/*
 * data_stream.cpp
 *
 *  created on: 2013-12-12
 *      Author: salmon
 */

extern "C"
{
#include <hdf5.h>
#include <hdf5_hl.h>

#ifdef USE_MPI
#include <mpi.h>
#endif
}

#include <cstring> //for memcopy

#include "hdf5_datatype.h"
#include "data_stream.h"
#include "../parallel/parallel.h"
#include "../parallel/message_comm.h"
#include "../parallel/mpi_datatype.h"
#include "../utilities/properties.h"
#include "../utilities/memory_pool.h"

#define H5_ERROR( _FUN_ ) if((_FUN_)<0){LOGGER<<"HDF5 Error:";H5Eprint(H5E_DEFAULT, stderr);}

namespace simpla
{

struct DataStream::pimpl_s
{
	hid_t file_;
	hid_t group_;

	struct DataSet
	{
		DataType data_desc;

		unsigned int ndims;

		hsize_t f_shape[MAX_NDIMS_OF_ARRAY];
		hsize_t f_offset[MAX_NDIMS_OF_ARRAY];
		hsize_t f_stride[MAX_NDIMS_OF_ARRAY];

		hsize_t m_shape[MAX_NDIMS_OF_ARRAY];
		hsize_t m_offset[MAX_NDIMS_OF_ARRAY];
		hsize_t m_stride[MAX_NDIMS_OF_ARRAY];

		hsize_t count[MAX_NDIMS_OF_ARRAY];
		hsize_t block[MAX_NDIMS_OF_ARRAY];

		unsigned int flag = 0UL;

	};

	Properties properties;

	typedef std::tuple<std::shared_ptr<ByteType>, DataSet> CacheDataSet;

	std::map<std::string, CacheDataSet> cache_;
	void open_group(std::string const & gname, unsigned int flag = 0UL);
	void open_file(std::string const &file_name = "unnamed", unsigned int flag = 0UL);
	void close_group();
	void close_file();
	void close();

	MemoryPool mempool_;
public:

	pimpl_s();
	~pimpl_s();

	bool is_ready()
	{
		return file_ > 0;
	}
	void set_property(std::string const & name, Any const&v)
	{
		properties[name] = v;
	}
	Any const & get_property_any(std::string const &name) const
	{
		return properties[name].template as<Any>();
	}

	void init(int argc = 0, char** argv = nullptr);

	bool command(std::string const & cmd);

	inline std::string pwd() const
	{
		return properties["File Name"].template as<std::string>() + ":"
		        + properties["Group Name"].template as<std::string>();
	}

	std::tuple<std::string, std::string> cd(std::string const &url_hint, unsigned int flag = 0UL);

	std::string write(std::string const &url, const void *, DataSet ds);

	/**
	 *
	 * @param res
	 * @param datatype
	 * @param rank_or_number
	 * @param global_begin
	 * @param global_end
	 * @param local_outer_begin
	 * @param local_outer_end
	 * @param local_inner_begin
	 * @param local_inner_end
	 * @return  DataSet
	 *
	 *  if global_begin ==nullptr and mpi is enable,
	 *      f_count[0] = sum( global_end[0], for all process)
	 *      f_begin[0] = sum( global_end[0], for process < this process)
	 *
	 *
	 *
	 */
	DataSet create_data_set(

	DataType const & datatype,

	size_t rank_or_number,

	size_t const *global_begin = nullptr,

	size_t const *global_end = nullptr,

	size_t const *local_outer_begin = nullptr,

	size_t const *local_outer_end = nullptr,

	size_t const *local_inner_begin = nullptr,

	size_t const *local_inner_end = nullptr,

	unsigned int flag = 0UL

	) const;

	void convert_record_data_set(DataSet*) const;

	std::string write_array(std::string const &name, const void *, DataSet const &);

	std::string write_cache(std::string const &name, const void *, DataSet const &);

	std::string flush_cache(std::string const & name);
};

DataStream::pimpl_s::pimpl_s() :
		file_(-1), group_(-1)
{
	hid_t error_stack = H5Eget_current_stack();
	H5Eset_auto(error_stack, NULL, NULL);

	properties["File Name"] = std::string("");

	properties["Group Name"] = std::string("/");

	properties["Suffix Width"] = 4;

	properties["Light Data Limit"] = 20;

	properties["Enable Compact Storage"] = false;

	properties["Enable XDMF"] = false;

	properties["Cache Depth"] = static_cast<int>(20);

}
DataStream::pimpl_s::~pimpl_s()
{
	close();
}

void DataStream::pimpl_s::init(int argc, char** argv)
{
	ParseCmdLine(argc, argv,

	[&,this](std::string const & opt,std::string const & value)->int
	{
		if(opt=="o"||opt=="output"||opt=="p"||opt=="prefix")
		{
			this->set_property("File Name",value);
		}
		else if(opt=="force-write-cache")
		{
			this->set_property("Force Write Cache",true);
		}
		else if(opt=="cache-depth")
		{
			this->set_property("Cache Depth",ToValue<size_t>(value));
		}
		return CONTINUE;
	}

	);

}

bool DataStream::pimpl_s::command(std::string const & cmd)
{
	if (cmd == "Flush")
	{
		for (auto const& item : cache_)
		{
			VERBOSE << flush_cache(item.first);
		}
	}
	return true;
}

void DataStream::pimpl_s::open_group(std::string const & gname, unsigned int)
{
	if (gname == "") return;

	hid_t h5fg = file_;

	close_group();

	std::string & grpname_ = properties["Group Name"].template as<std::string>();

	if (gname[0] == '/')
	{
		grpname_ = gname;
	}
	else
	{
		grpname_ += gname;
		if (group_ > 0) h5fg = group_;
	}

	if (grpname_[grpname_.size() - 1] != '/')
	{
		grpname_ = grpname_ + "/";
	}

	auto res = H5Lexists(h5fg, grpname_.c_str(), H5P_DEFAULT);

	if (grpname_ == "/" || res != 0)
	{
		H5_ERROR(group_ = H5Gopen(h5fg, grpname_.c_str(), H5P_DEFAULT));
	}
	else
	{
		H5_ERROR(group_ = H5Gcreate(h5fg, grpname_.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
	}
	if (group_ <= 0)
	{
		RUNTIME_ERROR("Can not open group " + grpname_ + " in file " + properties["Prefix"].template as<std::string>());
	}

	properties["Group Name"].template as<std::string>() = grpname_;

}

void sync_string(std::string * filename_)
{
#ifdef USE_MPI

	if (!GLOBAL_COMM.is_ready() )
	return;

	int name_len;

	if (GLOBAL_COMM.get_rank()==0) name_len=filename_->size();

	MPI_Bcast(&name_len, 1, MPI_INT, 0, GLOBAL_COMM.comm());

	std::vector<char> buffer(name_len);

	if (GLOBAL_COMM.get_rank()==0)
	{
		std::copy(filename_->begin(),filename_->end(),buffer.begin());
	}

	MPI_Bcast((&buffer[0]), name_len, MPI_CHAR, 0, GLOBAL_COMM.comm());

	buffer.push_back('\0');

	if (GLOBAL_COMM.get_rank()!=0)
	{
		*filename_=&buffer[0];
	}

#endif

}

void DataStream::pimpl_s::open_file(std::string const & fname, unsigned int flag)
{

	close_file();

	std::string file_name = (fname == "") ? "SIMPla_untitled.h5" : fname;

	hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

#ifdef USE_MPI
	H5Pset_fapl_mpio(plist_id, GLOBAL_COMM.comm(), GLOBAL_COMM.info());
#endif

	H5_ERROR(file_ = H5Fcreate( file_name.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id));

	H5Pclose(plist_id);

	if (file_ < 0)
	{
		RUNTIME_ERROR("create HDF5 file " + file_name + " failed!");
	}

	properties["File Name"].template as<std::string>() = file_name;

}

void DataStream::pimpl_s::close_group()
{
	if (group_ > 0)
	{
		H5Gclose(group_);
	}
	group_ = -1;
}
void DataStream::pimpl_s::close_file()
{
	close_group();

	if (file_ > 0)
	{
		H5Fclose(file_);
	}
	file_ = -1;
}
void DataStream::pimpl_s::close()
{
	close_group();
	close_file();
}

/**
 *
 * @param url_hint  <filename>:<group name>/<dataset name>
 * @param flag
 * @return
 */
std::tuple<std::string, std::string> DataStream::pimpl_s::cd(std::string const &url_hint, unsigned int flag)
{
//@todo using regex parser url
	std::string url = url_hint;

	std::string file_path(""), grp_name(""), dsname("");

	auto it = url_hint.find(':');

	if (it != std::string::npos)
	{
		file_path = url.substr(0, it);
		url = url.substr(it + 1);
	}

	it = url.rfind('/');

	if (it != std::string::npos)
	{
		grp_name = url.substr(0, it + 1);
		url = url.substr(it + 1);
	}

	if (url != "")
	{
		dsname = url;
	}

	auto current_file_path = properties["File Name"].template as<std::string>("");

	auto current_group_name = properties["Group Name"].template as<std::string>("/");

	if (file_path == "") file_path = current_file_path;

	if (grp_name == "") grp_name = current_group_name;

	if (file_ <= 0 || current_file_path != file_path)
	{

		if (GLOBAL_COMM.get_rank()==0)
		{
			std::string prefix = file_path;

			if (file_path.size() > 3 && file_path.substr(file_path.size() - 3) == ".h5")
			{
				prefix = file_path.substr(0, file_path.size() - 3);
			}

			/// @todo auto mkdir directory

			file_path = prefix +

			AutoIncrease(

			[&](std::string const & suffix)->bool
			{
				std::string fname=( prefix+suffix);
				return
				fname==""
				|| *(fname.rbegin())=='/'
				|| (CheckFileExists(fname + ".h5"));
			}

			) + ".h5";

		}
		sync_string(&file_path); // sync filename and open file

		open_file(file_path);
		open_group(grp_name);
	}
	else if (group_ <= 0 || current_group_name != grp_name)
	{
		open_group(grp_name);
	}

	if (dsname != "" && (flag & SP_APPEND) != SP_APPEND)
	{
		if (GLOBAL_COMM.get_rank() == 0)
		{
			dsname = dsname +

			AutoIncrease([&](std::string const & s )->bool
			{
				return H5Lexists(group_, (dsname + s ).c_str(), H5P_DEFAULT) > 0;
			}, 0, 4);
		}

		sync_string(&dsname);
	}

	return std::make_tuple(file_path + ":" + grp_name, dsname);
}
/**
 * @param pos in {0,count} out {begin,shape}
 */
std::tuple<hsize_t, hsize_t> sync_global_location(hsize_t count)
{
	hsize_t begin = 0;

#ifdef USE_MPI
	if ( GLOBAL_COMM.is_ready() && GLOBAL_COMM.get_size()>1 )
	{

		auto comm = GLOBAL_COMM.comm();

		int num_of_process = GLOBAL_COMM.get_size();
		int porcess_number = GLOBAL_COMM.get_rank();

		MPIDataType<hsize_t> m_type;

		std::vector<hsize_t> buffer;

		if (porcess_number == 0) buffer.resize(num_of_process);

		MPI_Gather(&count, 1, m_type.type(), &buffer[0], 1, m_type.type(), 0, comm);

		MPI_Barrier(comm);

		if (porcess_number == 0)
		{
			for (int i = 1; i < num_of_process; ++i)
			{
				buffer[i] += buffer[i - 1];
			}
			buffer[0] =count;
			count = buffer[num_of_process - 1];

			for (int i = num_of_process - 1; i > 0; --i)
			{
				buffer[i] = buffer[i - 1];
			}
			buffer[0] = 0;
		}
		MPI_Barrier(comm);
		MPI_Scatter(&buffer[0], 1, m_type.type(), &begin, 1, m_type.type(), 0, comm);
		MPI_Bcast(&count, 1, m_type.type(), 0, comm);
	}

#endif

	return std::make_tuple(begin, count);

}

std::string DataStream::pimpl_s::write(std::string const &url, void const* v, DataSet ds)
{

	if ((ds.flag & SP_RECORD) == SP_RECORD)
	{
		convert_record_data_set(&ds);
	}

	if ((ds.flag & SP_CACHE) == SP_CACHE)
	{
		return write_cache(url, v, ds);
	}
	else
	{
		return write_array(url, v, ds);
	}

}

DataStream::pimpl_s::DataSet DataStream::pimpl_s::create_data_set(

DataType const & data_desc,

size_t ndims,

size_t const *p_global_begin,

size_t const *p_global_end,

size_t const *p_local_outer_begin,

size_t const *p_local_outer_end,

size_t const *p_local_inner_begin,

size_t const *p_local_inner_end,

unsigned int flag) const
{
	DataSet res;

	res.data_desc = data_desc;

	res.flag = flag;

	for (int i = 0; i < ndims; ++i)
	{
		auto g_begin = (p_global_begin == nullptr) ? 0 : p_global_begin[i];

		res.f_shape[i] = (p_global_end == nullptr) ? 1 : p_global_end[i] - g_begin;

		res.f_stride[i] = res.f_shape[i];

		res.f_offset[i] = (p_local_inner_begin == nullptr) ? 0 : p_local_inner_begin[i] - g_begin;

		res.m_shape[i] =
		        (p_local_outer_end == nullptr || p_local_outer_begin == nullptr) ?
		                res.f_shape[i] : p_local_outer_end[i] - p_local_outer_begin[i];

		res.m_offset[i] =
		        (p_local_inner_begin == nullptr || p_local_outer_begin == nullptr) ?
		                0 : p_local_inner_begin[i] - p_local_outer_begin[i];

		res.m_stride[i] = res.m_shape[i];

		res.count[i] = 1;

		res.block[i] =
		        (p_local_inner_end == nullptr || p_local_inner_begin == nullptr) ?
		                res.f_shape[i] : p_local_inner_end[i] - p_local_inner_begin[i];

	}

	if ((flag & SP_UNORDER) == SP_UNORDER)
	{
		std::tie(res.f_offset[0], res.f_shape[0]) = sync_global_location(res.f_shape[0]);

		res.f_stride[0] = res.f_shape[0];
	}

	if (data_desc.NDIMS > 0)
	{
		for (int j = 0; j < data_desc.NDIMS; ++j)
		{

			res.f_shape[ndims + j] = data_desc.dimensions_[j];
			res.f_offset[ndims + j] = 0;
			res.f_stride[ndims + j] = res.f_shape[ndims + j];

			res.m_shape[ndims + j] = data_desc.dimensions_[j];
			res.m_offset[ndims + j] = 0;
			res.m_stride[ndims + j] = res.m_shape[ndims + j];

			res.count[ndims + j] = 1;
			res.block[ndims + j] = data_desc.dimensions_[j];

		}

		ndims += data_desc.NDIMS;
	}

	res.ndims = ndims;

	if (properties["Enable Compact Storage"].template as<bool>(false))
	{
		res.flag |= SP_APPEND;
	}

	if (properties["Force Record Storage"].template as<bool>(false))
	{
		res.flag |= SP_RECORD;
	}
	if (properties["Force Write Cache"].template as<bool>(false))
	{
		res.flag |= SP_CACHE;
	}
	return std::move(res);

}
void DataStream::pimpl_s::convert_record_data_set(DataSet *pds) const
{
	for (int i = pds->ndims; i > 0; --i)
	{

		pds->f_shape[i] = pds->f_shape[i - 1];
		pds->f_offset[i] = pds->f_offset[i - 1];
		pds->f_stride[i] = pds->f_stride[i - 1];
		pds->m_shape[i] = pds->m_shape[i - 1];
		pds->m_offset[i] = pds->m_offset[i - 1];
		pds->m_stride[i] = pds->m_stride[i - 1];
		pds->count[i] = pds->count[i - 1];
		pds->block[i] = pds->block[i - 1];

	}

	pds->f_shape[0] = 1;
	pds->f_offset[0] = 0;
	pds->f_stride[0] = 1;

	pds->m_shape[0] = 1;
	pds->m_offset[0] = 0;
	pds->m_stride[0] = 1;

	pds->count[0] = 1;
	pds->block[0] = 1;

	++pds->ndims;

}

std::string DataStream::pimpl_s::write_cache(std::string const & p_url, const void *v, DataSet const & ds)
{
	std::string path, dsname;

	std::tie(path, dsname) = cd(p_url, ds.flag);

	std::string url = path + dsname;

	if (cache_.find(url) == cache_.end())
	{
		size_t cache_memory_size = ds.data_desc.ele_size_in_byte_;
		for (int i = 0; i < ds.ndims; ++i)
		{
			cache_memory_size *= ds.m_shape[i];
		}

		size_t cache_depth = properties["Max Cache Size"].template as<size_t>(10 * 1024 * 1024UL) / cache_memory_size;

		if (cache_depth <= properties["Min Cache Number"].template as<int>(5))
		{
			return write_array(url, v, ds);
		}
		else
		{

			mempool_.allocate_shared_ptr<ByteType>(cache_memory_size * cache_depth).swap(std::get<0>(cache_[url]));

			DataSet & item = std::get<1>(cache_[url]);

			item.data_desc = ds.data_desc;

			item.flag = ds.flag | SP_APPEND;

			item.ndims = ds.ndims;

			for (int i = 0; i < ds.ndims; ++i)
			{

				item.f_shape[i] = ds.f_shape[i];

				item.f_offset[i] = ds.f_offset[i];

				item.f_stride[i] = ds.f_stride[i];

				item.m_shape[i] = ds.m_shape[i];

				item.m_offset[i] = ds.m_offset[i];

				item.m_stride[i] = ds.m_stride[i];

				item.count[i] = ds.count[i];

				item.block[i] = ds.block[i];

			}
			item.count[0] = 0;
			item.m_shape[0] = item.m_stride[0] * cache_depth + item.m_offset[0];
			item.f_shape[0] = item.f_stride[0] * cache_depth + item.f_offset[0];

		}
	}
	auto & data = std::get<0>(cache_[url]);
	auto & item = std::get<1>(cache_[url]);

	size_t memory_size = ds.data_desc.ele_size_in_byte_ * item.m_stride[0];

	for (int i = 1; i < item.ndims; ++i)
	{
		memory_size *= item.m_shape[i];
	}

	std::memcpy(reinterpret_cast<void*>(data.get() + item.count[0] * memory_size), v, memory_size);

	++item.count[0];

	if (item.count[0] * item.f_stride[0] + item.f_offset[0] >= item.m_shape[0])
	{
		return flush_cache(url);
	}
	else
	{
		return "\"" + url + "\" is write to cache";
	}

}
std::string DataStream::pimpl_s::flush_cache(std::string const & url)
{

	if (cache_.find(url) == cache_.end())
	{
		return url + " is not found !";
	}

	auto & data = std::get<0>(cache_[url]);
	auto & item = std::get<1>(cache_[url]);

	hsize_t t_f_shape = item.f_shape[0];
	hsize_t t_m_shape = item.m_shape[0];

	item.m_shape[0] = item.count[0] * item.m_stride[0] + item.m_offset[0];
	item.f_shape[0] = item.count[0] * item.f_stride[0] + item.f_offset[0];

	auto res = write_array(url, data.get(), item);

	item.m_shape[0] = t_f_shape;
	item.f_shape[0] = t_m_shape;

	item.count[0] = 0;

	return res;
}

std::string DataStream::pimpl_s::write_array(std::string const & p_url, const void *v, DataSet const &ds)
{
	if (v == nullptr)
	{
		WARNING << "Data is empty! Can not write to " << p_url;
		return "";
	}

	std::string path, dsname;

	std::tie(path, dsname) = cd(p_url, ds.flag);

	std::string url = path + dsname;

	hid_t m_type = GLOBAL_HDF5_DATA_TYPE_FACTORY.create( ds.data_desc.t_index_);

	hid_t file_space, mem_space;

	hid_t dset;

	if ((ds.flag & SP_APPEND) == 0)
	{

		file_space = H5Screate_simple(ds.ndims, ds.f_shape, nullptr);

		dset = H5Dcreate(group_, dsname.c_str(), m_type, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

		H5_ERROR(H5Sclose(file_space));

		H5_ERROR(H5Fflush(group_, H5F_SCOPE_GLOBAL));

		file_space = H5Dget_space(dset);

		H5_ERROR(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, ds.f_offset, ds.f_stride, ds.count, ds.block));

		mem_space = H5Screate_simple(ds.ndims, ds.m_shape, NULL);

		H5_ERROR(H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, ds.m_offset, ds.m_stride, ds.count, ds.block));

	}
	else
	{
		if (H5Lexists(group_, dsname.c_str(), H5P_DEFAULT) == 0)
		{
			int f_ndims = ds.ndims;
			hsize_t current_dims[MAX_NDIMS_OF_ARRAY];
			hsize_t maximum_dims[MAX_NDIMS_OF_ARRAY];

			std::copy(ds.f_shape, ds.f_shape + ds.ndims, current_dims);

			std::copy(ds.f_shape, ds.f_shape + ds.ndims, maximum_dims);

			hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);

			H5_ERROR(H5Pset_chunk(dcpl_id, f_ndims, current_dims));

			maximum_dims[0] = H5S_UNLIMITED;

			current_dims[0] = 0;

			hid_t f_space = H5Screate_simple(f_ndims, current_dims, maximum_dims);

			hid_t t_dset = H5Dcreate(group_, dsname.c_str(), m_type, f_space, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

			H5_ERROR(H5Sclose(f_space));

			H5_ERROR(H5Dclose(t_dset));

			H5_ERROR(H5Pclose(dcpl_id));

		}

		dset = H5Dopen(group_, dsname.c_str(), H5P_DEFAULT);

		file_space = H5Dget_space(dset);

		int ndims = H5Sget_simple_extent_ndims(file_space);

		hsize_t f_shape[MAX_NDIMS_OF_ARRAY];
		hsize_t f_offset[MAX_NDIMS_OF_ARRAY];

		H5Sget_simple_extent_dims(file_space, f_shape, nullptr);

		H5Sclose(file_space);

		std::copy(ds.f_offset, ds.f_offset + ndims, f_offset);

		f_offset[0] += f_shape[0];

		f_shape[0] += ds.f_shape[0];

		H5Dset_extent(dset, f_shape);

		file_space = H5Dget_space(dset);

		H5_ERROR(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, f_offset, ds.f_stride, ds.count, ds.block));

		mem_space = H5Screate_simple(ds.ndims, ds.m_shape, nullptr);

		H5_ERROR(H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, ds.m_offset, ds.m_stride, ds.count, ds.block));

		//	CHECK(ndims);
		//	CHECK(f_shape[0]) << " " << f_shape[1];
		//	CHECK(ds.f_offset[0]) << " " << ds.f_offset[1];
		//	CHECK(ds.m_shape[0]) << " " << ds.m_shape[1];
		//	CHECK(ds.m_offset[0]) << " " << ds.m_offset[1];
		//	CHECK(ds.m_stride[0]) << " " << ds.m_stride[1];
		//	CHECK(ds.count[0]) << " " << ds.count[1];
		//	CHECK(ds.block[0]) << " " << ds.block[1];
	}

// create property list for collective DataSet write.
#ifdef USE_MPI
	hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5_ERROR(H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT));
	H5_ERROR(H5Dwrite(dset, m_type, mem_space, file_space, plist_id, v));
	H5_ERROR(H5Pclose(plist_id));
#else
	H5_ERROR(H5Dwrite(dset,m_type , mem_space, file_space, H5P_DEFAULT, v));
#endif

	H5_ERROR(H5Dclose(dset));

	H5_ERROR(H5Sclose(mem_space));

	H5_ERROR(H5Sclose(file_space));

	if (H5Tcommitted(m_type) > 0) H5Tclose(m_type);

	return url;
}

//=====================================================================================
DataStream::DataStream() :
		pimpl_(new pimpl_s)
{
}
DataStream::~DataStream()
{
}
bool DataStream::is_ready() const
{
	return pimpl_->is_ready();
}
void DataStream::init(int argc, char** argv)
{
	pimpl_->init(argc, argv);
}
std::string DataStream::cd(std::string const & gname, unsigned int flag)
{
	return std::get<0>(pimpl_->cd(gname));
}

void DataStream::set_property_(std::string const & name, Any const &v)
{
	pimpl_->set_property(name, v);
}
Any DataStream::get_property_(std::string const & name) const
{
	return pimpl_->get_property_any(name);
}

std::string DataStream::pwd() const
{
	return pimpl_->pwd();
}
void DataStream::close()
{
	return pimpl_->close();
}
std::string DataStream::write(std::string const &name, void const *v,

DataType const & data_desc,

size_t ndims_or_number,

size_t const *global_begin,

size_t const *global_end,

size_t const *local_outer_begin,

size_t const *local_outer_end,

size_t const *local_inner_begin,

size_t const *local_inner_end,

unsigned int flag

)
{
	return pimpl_->write(name, v,

	pimpl_->create_data_set(data_desc, ndims_or_number,

	global_begin, global_end,

	local_outer_begin, local_outer_end,

	local_inner_begin, local_inner_end,

	flag));
}
bool DataStream::command(std::string const & cmd)
{
	return pimpl_->command(cmd);
}
}
// namespace simpla
