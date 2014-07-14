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

#include "hdf5_datatype.h"
#include "data_stream.h"
#include "../parallel/parallel.h"
#include "../parallel/message_comm.h"
#include "../parallel/mpi_datatype.h"
#include "../utilities/properties.h"
#include "../utilities/memory_pool.h"

#define H5_ERROR( _FUN_ ) if((_FUN_)<0){ H5Eprint(H5E_DEFAULT, stderr);}

namespace simpla
{

struct DataStream::pimpl_s
{
	hid_t file_;
	hid_t group_;

	Properties properties;

	struct DataSet
	{

		DataType data_desc;

		unsigned int ndims;

		hsize_t f_shape[MAX_NDIMS_OF_ARRAY];
		hsize_t f_begin[MAX_NDIMS_OF_ARRAY];
		hsize_t m_shape[MAX_NDIMS_OF_ARRAY];
		hsize_t m_begin[MAX_NDIMS_OF_ARRAY];
		hsize_t m_count[MAX_NDIMS_OF_ARRAY];

		unsigned int flag = 0UL;
	};

	typedef std::pair<std::shared_ptr<ByteType>, DataSet> CacheDataSet;

	std::map<std::string, CacheDataSet> cache_;

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
	void open_group(std::string const & gname, unsigned int flag = 0UL);
	void open_file(std::string const &fname = "unnamed", unsigned int flag = 0UL);
	void close_group();
	void close_file();
	void close();

	inline std::string get_current_path() const
	{
		return properties["File Name"].template as<std::string>() + ":"
		        + properties["Group Name"].template as<std::string>();
	}

	template<typename ... Args>
	std::string write(std::string const &name, const void *, Args &&... args);

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
	 * @return  if data has struct (global_end!=nullptr) return true ,else return  false
	 */
	bool create_data_set(DataSet*res,

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

	std::string write_array(std::string const &name, const void *, DataSet const &);

	std::string write_unorder_data(std::string const &name, const void *, DataSet const &);

	std::string write_cache(std::string const &name, const void *, DataSet const &);
};

DataStream::pimpl_s::pimpl_s() :
		file_(-1), group_(-1)
{
	hid_t error_stack = H5Eget_current_stack();
	H5Eset_auto(error_stack, NULL, NULL);

	properties["Prefix"] = std::string("simpla_unnamed");

	properties["File Name"] = std::string("unnamed");

	properties["Group Name"] = std::string("");

	properties["Suffix Width"] = 4;

	properties["Light Data Limit"] = 20;

	properties["Enable Compact Storage"] = false;

	properties["Enable XDMF"] = false;

	properties["Cache Size In Bytes"] = static_cast<size_t>(1024 * 1024);

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
			this->open_file(value);
		}
		return CONTINUE;
	}

	);

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

void DataStream::pimpl_s::open_file(std::string const &fname, unsigned int flag)
{

	close_file();

	std::string filename_ = properties["File Name"].template as<std::string>();
	std::string & prefix = properties["Prefix"].template as<std::string>();
	GLOBAL_COMM.barrier();
	if (GLOBAL_COMM.get_rank()==0)
	{
		if (fname != "")
		prefix = fname;

		if (fname.size() > 3 && fname.substr(fname.size() - 3) == ".h5")
		{
			prefix = fname.substr(0, fname.size() - 3);
		}

		/// @todo auto mkdir directory

		filename_ = prefix +

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

	hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

	if ((flag & SP_CACHE) > 0)
	{
		// @todo fixme ,this is nothing
		H5Pset_cache(plist_id, 1 /*mdc_nelmts*/, 1 /* rdcc_nbytes*/, 1, 1.0 /*w0*/);
	}

#ifdef USE_MPI

	if (GLOBAL_COMM.is_ready())
	{
		sync_string(& filename_); // sync filename and open file
		H5Pset_fapl_mpio(plist_id, GLOBAL_COMM.comm(), GLOBAL_COMM.info());

	}
#endif

	H5_ERROR(file_ = H5Fcreate( filename_.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id));

	H5Pclose(plist_id);

	GLOBAL_COMM.barrier();

	if (file_ < 0)
	{
		RUNTIME_ERROR("create HDF5 file " + filename_ + " failed!");
	}

	properties["File Name"].template as<std::string>() = filename_;
	open_group("/");
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

bool DataStream::pimpl_s::create_data_set(DataSet*res,

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

	res->data_desc = data_desc;

	res->ndims = ndims;

	res->flag = flag;

	if (p_global_end == nullptr)
	{
		return false;
	}
	else
	{
		for (int i = 0; i < ndims; ++i)
		{
			auto g_begin = (p_global_begin == nullptr) ? 0 : p_global_begin[i];

			res->f_shape[i] = (p_global_end == nullptr) ? 1 : p_global_end[i] - g_begin;

			res->f_begin[i] = (p_local_inner_begin == nullptr) ? 0 : p_local_inner_begin[i] - g_begin;

			res->m_shape[i] =
			        (p_local_outer_end == nullptr || p_local_outer_begin == nullptr) ?
			                res->f_shape[i] : p_local_outer_end[i] - p_local_outer_begin[i];

			res->m_begin[i] =
			        (p_local_inner_begin == nullptr || p_local_outer_begin == nullptr) ?
			                0 : p_local_inner_begin[i] - p_local_outer_begin[i];

			res->m_count[i] =
			        (p_local_inner_end == nullptr || p_local_inner_begin == nullptr) ?
			                res->f_shape[i] : p_local_inner_end[i] - p_local_inner_begin[i];
		}

		if (data_desc.NDIMS > 0)
		{
			for (int j = 0; j < data_desc.NDIMS; ++j)
			{

				res->f_shape[ndims + j] = data_desc.dimensions_[j];
				res->f_begin[ndims + j] = 0;
				res->m_shape[ndims + j] = data_desc.dimensions_[j];
				res->m_begin[ndims + j] = 0;
				res->m_count[ndims + j] = data_desc.dimensions_[j];

				++ndims;
			}
		}
		res->ndims = ndims;
	}
	return true;

}
template<typename ...Args>
std::string DataStream::pimpl_s::write(std::string const &name, const void *v, Args &&... args)
{

	DataSet ds;

	if (create_data_set(&ds, std::forward<Args>(args)...))
	{
		if ((ds.flag & SP_CACHE) > 0)
		{
			return write_cache(name, v, ds);
		}
		else
		{
			return write_array(name, v, ds);
		}
	}
	else
	{
		return write_unorder_data(name, v, ds);

	}

}

void calc_strides(bool is_fast_first, unsigned int ndims, hsize_t const &count[], hsize_t * strides)
{

}
std::string DataStream::pimpl_s::write_cache(std::string const & name, const void *v, DataSet const & ds)
{
	size_t cache_depth = properties["Cache Depth"].as<size_t>(1000UL);

	std::string url = get_current_path() + "/" + name;

	unsigned int ndims = ds.ndims;
	unsigned int ele_size_in_byte = ds.data_desc.ele_size_in_byte_;

	auto data = std::get<0>(cache_[url]);
	auto & item = std::get<1>(cache_[url]);

	if (data == nullptr)
	{
		item.ndims = ds.ndims;
		item.data_desc = ds.data_desc;
		item.flag = ds.flag;

		size_t cache_memory_size = cache_depth * ds.data_desc.ele_size_in_byte_;

		for (int i = 0; i < ndims; ++i)
		{
			cache_memory_size *= ds.m_count[i];

			item.f_shape[i] = ds.f_shape[i];
			item.f_begin[i] = ds.f_begin[i];
			item.m_begin[i] = 0;
			item.m_shape[i] = ds.m_count[i];
			item.m_count[i] = ds.m_count[i];
		}

		if ((item.flag & SP_APPEND) > 0)
		{
			item.f_shape[0] *= cache_depth;
			item.m_shape[0] *= cache_depth;
		}
		else
		{
			item.f_shape[ndims] = cache_depth;
			item.f_begin[ndims] = 0;
			item.m_shape[ndims] = cache_depth;
			item.m_begin[ndims] = 0;
			item.m_count[ndims] = 1;
			++item.ndims;
		}
		data = MEMPOOL.allocate_shared_ptr< ByteType> (cache_memory_size);
	}

	hsize_t m_strides[MAX_NDIMS_OF_ARRAY];
	hsize_t f_strides[MAX_NDIMS_OF_ARRAY];

	calc_strides((ds.flag & SP_FAST_FIRST) > 0, ds.ndims, ds.m_shape, m_strides);
	calc_strides((item.flag & SP_FAST_FIRST) > 0, item.ndims, item.m_shape, m_strides);

	hsize_t idx[MAX_NDIMS_OF_ARRAY];

	for (int i = 0; i < ndims; ++i)
	{
		idx[i] = 0;
	}

	while (idx[0] < ds.m_count[0])
	{
		hsize_t f_pos = 0;
		hsize_t m_pos = 0;

		for (int i = 0; i < ndims; ++i)
		{
			f_pos += (idx[i] + item.m_count[i] - ds.m_count[i]) * f_strides[i];
			m_pos += (idx[i] + ds.m_begin[i]) * m_strides[i];
		}
		for (int i = 0; i < ele_size_in_byte; ++i)
		{
			(data.get() + f_pos * ele_size_in_byte + i) = reinterpret_cast<ByteType*>(v) + m_pos * ele_size_in_byte + i;
		}
		++idx[ndims - 1];
		for (int i = ndims - 1; i > 0; --i)
		{
			if (idx[i] >= ds.m_count[i])
			{
				idx[i] = 0;
				++idx[i - 1];
			}
		}
	}

	if ((item.flag & SP_APPEND) > 0)
	{
		item.m_count[0] += ds.m_count[0];

		if ((item.m_count[0] >= item.m_shape[0]))
		{
			write_array(name, data.get(), item);
			item.m_count[0] = 0;
		}
	}
	else
	{
		++item.m_count[item.ndims];

		if (item.m_count[item.ndims] >= item.m_shape[item.ndims])
		{
			write_array(name, data.get(), item);
			item.m_count[item.ndims] = 0;
		}
	}

//	size_t size = 1024 * 1024;
//
//	if (prop_["Cache Size In Bytes"])
//	{
//		size = prop_["Cache Size In Bytes"].template as<size_t>();
//	}
//
//	cache_s item;
//
//	datadesc = data_desc;
//
//	item.ndims = ndims;
//
//	for (int i = 0; i < ndims; ++i)
//	{
//	item.dims[i]=
//}
//
//size_t dims[MAX_NDIMS_OF_ARRAY];
//
//item.data_ = MEMPOOL.allocate_shared_ptr< ByteType> (num_of_ele_);
//
//	size_t tail_;
//
//	MEMPOOL.allocate_shared_ptr< ByteType> (num_of_ele_);

	UNIMPLEMENT;
	return "\"" + url + "\" is write to cache";

}

std::string DataStream::pimpl_s::write_array(std::string const & name, const void *v, DataSet const &p_ds)
{
	DataSet ds = p_ds;

	assert((flag&SP_FAST_FIRST) == 0UL);	/// @todo add support for FASF_FIRST array

	std::string dsname = name;

	if (v == nullptr)
	{
		WARNING << dsname << " is empty!";
		return "";
	}

	if (group_ <= 0)
	{
		WARNING << "HDF5 file is not opened! No data is saved!";
	}

	hid_t m_type = GLOBAL_HDF5_DATA_TYPE_FACTORY.create(ds.data_desc.t_index_);

	hid_t dset;

	hid_t file_space, mem_space;

	if (!properties["Enable Compact Storage"].template as<bool>())
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

		file_space = H5Screate_simple(ds.ndims, ds.f_shape, nullptr);

		dset = H5Dcreate(group_, dsname.c_str(), m_type, file_space,

		H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

		H5_ERROR(H5Sclose(file_space));

		H5_ERROR(H5Fflush(group_, H5F_SCOPE_GLOBAL));

		file_space = H5Dget_space(dset);

		H5_ERROR(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, ds.f_begin, NULL, ds.m_count, NULL));

		mem_space = H5Screate_simple(ds.ndims, ds.m_shape, NULL);

		H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, ds.m_begin, NULL, ds.m_count, NULL);

	}
	else if (!(ds.flag & SP_APPEND)) // add new record, extent the 'rank' dimension; file data has rank+1 dimension
	{

		ds.f_shape[ds.ndims] = 1;
		ds.f_begin[ds.ndims] = 0;
		ds.m_shape[ds.ndims] = 1;
		ds.m_begin[ds.ndims] = 0;
		ds.m_count[ds.ndims] = 0;

		if (H5Lexists(group_, dsname.c_str(), H5P_DEFAULT) == 0)
		{
			hsize_t max_dims[ds.ndims + 1];

			std::copy(ds.f_shape, ds.f_shape + ds.ndims, max_dims);

			max_dims[ds.ndims] = H5S_UNLIMITED;

			hid_t space = H5Screate_simple(ds.ndims + 1, ds.f_shape, max_dims);

			hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);

			H5_ERROR(H5Pset_chunk(dcpl_id, ds.ndims + 1, ds.f_shape));

			dset = H5Dcreate(group_, dsname.c_str(), m_type, space, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

			H5_ERROR(H5Sclose(space));

			H5_ERROR(H5Pclose(dcpl_id));

		}
		else
		{

			dset = H5Dopen(group_, dsname.c_str(), H5P_DEFAULT);

			file_space = H5Dget_space(dset);

			H5Sget_simple_extent_dims(file_space, ds.f_shape, nullptr);

			H5Sclose(file_space);

			++ds.f_shape[ds.ndims];

			H5Dset_extent(dset, ds.f_shape);
		}

		file_space = H5Dget_space(dset);

		H5_ERROR(H5Sget_simple_extent_dims(file_space, ds.f_shape, nullptr));

		ds.f_begin[ds.ndims] = ds.f_shape[ds.ndims] - 1;

		ds.m_count[ds.ndims] = 1;

		H5_ERROR(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, ds.f_begin, nullptr, ds.m_count, nullptr));

		mem_space = H5Screate_simple(ds.ndims + 1, ds.m_shape, nullptr);

		H5_ERROR(H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, ds.m_begin, NULL, ds.m_count, NULL));

	}
	else //   append, extent first dimension; file data has rank dimension
	{

		if (H5Lexists(group_, dsname.c_str(), H5P_DEFAULT) == 0)
		{
			hsize_t max_dims[ds.ndims];

			std::copy(ds.f_shape, ds.f_shape + ds.ndims, max_dims);

			max_dims[0] = H5S_UNLIMITED;

			hid_t space = H5Screate_simple(ds.ndims, ds.f_shape, max_dims);

			hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);

			H5_ERROR(H5Pset_chunk(dcpl_id, ds.ndims, ds.f_shape));

			dset = H5Dcreate(group_, dsname.c_str(), m_type, space, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

			H5_ERROR(H5Sclose(space));

			H5_ERROR(H5Pclose(dcpl_id));

		}
		else
		{

			dset = H5Dopen(group_, dsname.c_str(), H5P_DEFAULT);

			file_space = H5Dget_space(dset);

			int ndims = H5Sget_simple_extent_ndims(file_space);

			assert(ndims==ndims);

			H5Sget_simple_extent_dims(file_space, ds.f_shape, nullptr);

			H5Sclose(file_space);

			ds.f_shape[0] += ds.m_count[0];

			H5Dset_extent(dset, ds.f_shape);
		}

		//		CHECK(name);
		//		CHECK(rank);
		//		CHECK(ds.g_shape[0]) << " " << ds.g_shape[1] << " " << ds.g_shape[2] << " " << ds.g_shape[3];
		//		CHECK(ds.f_begin[0]) << " " << ds.f_begin[1] << " " << ds.f_begin[2] << " " << ds.f_begin[3];
		//		CHECK(ds.m_shape[0]) << " " << ds.m_shape[1] << " " << ds.m_shape[2] << " " << ds.m_shape[3];
		//		CHECK(m_begin[0]) << " " << m_begin[1] << " " << m_begin[2] << " " << m_begin[3];
		//		CHECK(m_count[0]) << " " << m_count[1] << " " << m_count[2] << " " << m_count[3];

		file_space = H5Dget_space(dset);

		H5_ERROR(H5Sget_simple_extent_dims(file_space, ds.f_shape, nullptr));

		ds.f_begin[0] = ds.f_shape[0] - ds.m_count[0];

		H5_ERROR(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, ds.f_begin, nullptr, ds.m_count, nullptr));

		mem_space = H5Screate_simple(ds.ndims, ds.m_shape, nullptr);

		H5_ERROR(H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, ds.m_begin, NULL, ds.m_count, NULL));

	}

// create property list for collective DataSet write.

#ifdef USE_MPI
	if (GLOBAL_COMM.is_ready())
	{
		hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
		H5_ERROR(H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT));
		H5_ERROR(H5Dwrite(dset, m_type, mem_space, file_space, plist_id, v));
		H5_ERROR(H5Pclose(plist_id));
	}
	else
#endif
	{
		H5_ERROR(H5Dwrite(dset,m_type , mem_space, file_space, H5P_DEFAULT, v));
	}

	H5_ERROR(H5Dclose(dset));

	H5_ERROR(H5Sclose(mem_space));

	H5_ERROR(H5Sclose(file_space));

	if (H5Tcommitted(m_type) > 0) H5Tclose(m_type);

	return "\"" + get_current_path() + dsname + "\"";
}

void sync_location(hsize_t count[2])
{

#ifdef USE_MPI

	if (!GLOBAL_COMM.is_ready()) return;

	auto comm = GLOBAL_COMM.comm();

	int size = GLOBAL_COMM.get_size();
	int rank = GLOBAL_COMM.get_rank();

	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	if (size <= 1)
	{
		return;
	}

	MPIDataType<hsize_t> m_type;

	std::vector<hsize_t> buffer;

	if (rank == 0) buffer.resize(size);

	MPI_Gather(&count[1], 1, m_type.type(), &buffer[0], 1, m_type.type(), 0, comm);

	MPI_Barrier(comm);
	if (rank == 0)
	{
		for (int i = 1; i < size; ++i)
		{
			buffer[i] += buffer[i - 1];
		}
		buffer[0] = count[1];
		count[1] = buffer[size - 1];

		for (int i = size - 1; i > 0; --i)
		{
			buffer[i] = buffer[i - 1];
		}
		buffer[0] = 0;
	}
	MPI_Barrier(comm);
	MPI_Scatter(&buffer[0], 1, m_type.type(), &count[0], 1, m_type.type(), 0, comm);
	MPI_Bcast(&count[1], 1, m_type.type(), 0, comm);
#endif
}

std::string DataStream::pimpl_s::write_unorder_data(std::string const &name, const void *v, DataSet const &ds)
{

	auto dsname = name;

	if (v == nullptr)
	{
		WARNING << dsname << " is empty!";
		return "";
	}

	if (group_ <= 0)
	{
		WARNING << "HDF5 file is not opened! No data is saved!";
	}

	hsize_t pos[2] = { 0, ds.ndims };

	sync_location(pos);

	int rank = ds.data_desc.NDIMS + 1;

	hsize_t f_count[rank];
	hsize_t f_begin[rank];
	hsize_t m_count[rank];

	f_begin[0] = pos[0];
	f_count[0] = pos[1];
	m_count[0] = ds.ndims;

	if (ds.data_desc.NDIMS > 0)
	{
		for (int j = 0; j < ds.data_desc.NDIMS; ++j)
		{

			f_count[1 + j] = ds.data_desc.dimensions_[j];

			f_begin[1 + j] = 0;

			m_count[1 + j] = ds.data_desc.dimensions_[j];

		}
	}

	hid_t m_type = GLOBAL_HDF5_DATA_TYPE_FACTORY.create(ds.data_desc.t_index_);

	hid_t dset;

	hid_t file_space, mem_space;

	dsname = dsname +

	AutoIncrease([&](std::string const & s )->bool
	{
		return H5Lexists(group_, (dsname + s ).c_str(), H5P_DEFAULT) > 0;
	}, 0, 4);

	file_space = H5Screate_simple(rank, f_count, nullptr);

	dset = H5Dcreate(group_, dsname.c_str(), m_type, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	H5_ERROR(H5Sclose(file_space));

	H5_ERROR(H5Fflush(group_, H5F_SCOPE_GLOBAL));

	file_space = H5Dget_space(dset);

	H5_ERROR(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, f_begin, NULL, m_count, NULL));

	mem_space = H5Screate_simple(rank, m_count, NULL);

// create property list for collective data set write.
	if (GLOBAL_COMM.is_ready())
	{
		hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
		H5_ERROR(H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT));
		H5_ERROR(H5Dwrite(dset, m_type, mem_space, file_space, plist_id, v));
		H5_ERROR(H5Pclose(plist_id));
	}
	else
	{
		H5_ERROR(H5Dwrite(dset,m_type , mem_space, file_space, H5P_DEFAULT, v));
	}

	H5_ERROR(H5Dclose(dset));

	H5_ERROR(H5Sclose(mem_space));

	H5_ERROR(H5Sclose(file_space));

	if (H5Tcommitted(m_type) > 0) H5Tclose(m_type);

	return "\"" + get_current_path() + dsname + "\"";
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
void DataStream::open_group(std::string const & gname)
{
	pimpl_->open_group(gname);
}
void DataStream::open_file(std::string const &fname)
{
	pimpl_->open_file(fname);
}
void DataStream::close_group()
{
	pimpl_->close_group();
}
void DataStream::close_file()
{
	pimpl_->close_file();
}
void DataStream::close()
{
	pimpl_->close();
}

void DataStream::set_property_(std::string const & name, Any const &v)
{
	pimpl_->set_property(name, v);
}
Any DataStream::get_property_(std::string const & name) const
{
	return pimpl_->get_property_any(name);
}

std::string DataStream::get_current_path() const
{
	return pimpl_->get_current_path();
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
	return pimpl_->write(name,

	v,

	data_desc,

	ndims_or_number,

	global_begin,

	global_end,

	local_outer_begin,

	local_outer_end,

	local_inner_begin,

	local_inner_end,

	flag);
}
}
// namespace simpla
