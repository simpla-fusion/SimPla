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

}

#include <cstring> //for memcopy

#include "data_stream.h"
#include "../data_structure/data_set.h"

#ifdef USE_MPI
#include "../parallel/parallel.h"
#include "../parallel/message_comm.h"
#include "../parallel/mpi_aux_functions.h"
#endif

#include "../utilities/properties.h"
#include "../utilities/memory_pool.h"
#include "../utilities/parse_command_line.h"

#define H5_ERROR( _FUN_ ) if((_FUN_)<0){LOGGER<<"HDF5 Error:";H5Eprint(H5E_DEFAULT, stderr);}

namespace simpla
{

struct DataStream::pimpl_s
{
	hid_t base_file_id_;
	hid_t base_group_id_;

	std::string current_groupname_;
	std::string current_filename_;

	struct h5_dataset
	{
		DataType datatype;

		size_t ndims;

		hsize_t f_dims[MAX_NDIMS_OF_ARRAY];
		hsize_t f_offset[MAX_NDIMS_OF_ARRAY];
		hsize_t f_stride[MAX_NDIMS_OF_ARRAY];

		hsize_t m_dims[MAX_NDIMS_OF_ARRAY];
		hsize_t m_offset[MAX_NDIMS_OF_ARRAY];
		hsize_t m_stride[MAX_NDIMS_OF_ARRAY];

		hsize_t count[MAX_NDIMS_OF_ARRAY];
		hsize_t block[MAX_NDIMS_OF_ARRAY];

		size_t flag = 0UL;

	};

	typedef std::tuple<std::shared_ptr<ByteType>, h5_dataset> CacheDataSet;

	std::map<std::string, CacheDataSet> cache_;

	std::tuple<std::string, hid_t> open_group(std::string const & path);
	std::tuple<std::string, hid_t> open_file(std::string const & path,
			bool is_append = false);

	void close();

	void flush_all();

	MemoryPool mempool_;
public:

	Properties properties;

	pimpl_s();
	~pimpl_s();

	bool is_valid()
	{
		return base_file_id_ > 0;
	}

	void init(int argc = 0, char** argv = nullptr);

	bool command(std::string const & cmd);

	std::string pwd() const
	{
		return current_filename_ + ":" + current_groupname_;
//		properties["File Name"].as<std::string>() + ":" + properties["Group Name"].as<std::string>();
	}

	bool check_null_dataset(h5_dataset const & ds)
	{
		bool is_null = true;

		size_t s = 1;
		for (int i = 0; i < ds.ndims; ++i)
		{
			s *= (ds.m_dims[i]);
		}
		return s == 0;
	}

	std::string cd(std::string const &file_name, std::string const &grp_name,
			size_t is_append = 0UL);

	std::string cd(std::string const &url, size_t is_append = 0UL);

	std::string write(std::string const &url, const void *, h5_dataset ds);

	std::string write(std::string const &url, DataSet const &ds, size_t flag =
			0UL);
	std::string read(std::string const &url, DataSet * ds, size_t flag = 0UL);

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
	h5_dataset create_h5_dataset(

	DataType const & datatype,

	size_t rank_or_number,

	size_t const *global_begin = nullptr,

	size_t const *global_end = nullptr,

	size_t const *local_outer_begin = nullptr,

	size_t const *local_outer_end = nullptr,

	size_t const *local_inner_begin = nullptr,

	size_t const *local_inner_end = nullptr,

	size_t flag = 0UL

	) const;

	h5_dataset create_h5_dataset(DataSet const & ds, size_t flag = 0UL) const;

	void convert_record_dataset(h5_dataset*) const;

	std::string write_array(std::string const &name, const void *,
			h5_dataset const &);

	std::string write_cache(std::string const &name, const void *,
			h5_dataset const &);

	std::string flush_cache(std::string const & name);

	hid_t create_h5_datatype(DataType const &, bool is_compact_array = false);

	void set_attribute(std::string const &url, DataType const &d_type,
			void const * buff);

	void get_attribute(std::string const &url, DataType const &d_type,
			void *buff);

	void delete_attribute(std::string const &url);

	void delete_attribute(std::string const &obj_name,
			std::string const & attr_name);

	std::tuple<std::string, std::string, std::string, std::string> parser_url(
			std::string const & url);

};

DataStream::pimpl_s::pimpl_s() :
		base_file_id_(-1), base_group_id_(-1), current_filename_("untitle.h5"), current_groupname_(
				"/")
{
	hid_t error_stack = H5Eget_current_stack();
	H5Eset_auto(error_stack, NULL, NULL);

	properties["File Name"] = std::string("");

	properties["Group Name"] = std::string("/");

	properties["Suffix Width"] = 4;

	properties["Light Data Limit"] = 20;

	properties["Enable Compact Storage"] = false;

	properties["Enable XDMF"] = false;

	properties["Cache Depth"] = static_cast<int>(50);

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
			properties.set("File Name",value);
		}
		else if(opt=="force-write-cache")
		{
			properties.set("Force Write Cache",true);
		}
		else if(opt=="cache-depth")
		{
			properties.set("Cache Depth",ToValue<size_t>(value));
		}
		return CONTINUE;
	}

	);

	current_filename_ = properties["File Name"].template as<std::string>();
	current_groupname_ = "/";
	cd(pwd());

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

void sync_string(std::string * filename_)
{
#ifdef USE_MPI

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

std::tuple<std::string, hid_t> DataStream::pimpl_s::open_file(
		std::string const & fname, bool is_append)
{
	std::string filename = fname;

	if (filename == "")
		filename = current_filename_;

	if (!is_append)
	{
#ifdef USE_MPI
		if (GLOBAL_COMM.get_rank() == 0)
#endif
		{
			std::string prefix = filename;

			if (filename.size() > 3
					&& filename.substr(filename.size() - 3) == ".h5")
			{
				prefix = filename.substr(0, filename.size() - 3);
			}

			/// @todo auto mkdir directory

			filename = prefix +

			AutoIncrease(

			[&](std::string const & suffix)->bool
			{
				std::string f=( prefix+suffix);
				return
				f==""
				|| *(f.rbegin())=='/'
				|| (CheckFileExists(f + ".h5"));
			}

			) + ".h5";

		}
	}

	hid_t f_id;

	hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

#ifdef USE_MPI
	H5Pset_fapl_mpio(plist_id, GLOBAL_COMM.comm(), GLOBAL_COMM.info());
#endif

	H5_ERROR(
			f_id = H5Fcreate( filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id));

	H5Pclose(plist_id);

	return std::make_tuple(filename, f_id);

}

std::tuple<std::string, hid_t> DataStream::pimpl_s::open_group(
		std::string const & str)
{
	std::string path = str;
	hid_t g_id = -1;

	if (path[0] != '/')
	{
		path = current_groupname_ + path;
	}

	if (path[path.size() - 1] != '/')
	{
		path = path + "/";
	}

	if (path == "/" || H5Lexists(base_file_id_, path.c_str(), H5P_DEFAULT) != 0)
	{
		H5_ERROR(g_id = H5Gopen(base_file_id_, path.c_str(), H5P_DEFAULT));
	}
	else
	{
		H5_ERROR(
				g_id = H5Gcreate(base_file_id_, path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
	}

	return std::make_tuple(path, g_id);

}

/**
 *
 * @param url_hint  <filename>:<group name>/<dataset name>
 * @param flag
 * @return
 */
std::string DataStream::pimpl_s::cd(std::string const &url, size_t is_append)
{
	std::string file_name, grp_name, obj_name;
	std::tie(file_name, grp_name, obj_name, std::ignore) = parser_url(url);
	return cd(file_name, grp_name + obj_name, is_append);
}

std::string DataStream::pimpl_s::cd(std::string const &file_name,
		std::string const &grp_name, size_t is_append)
{
//@todo using regex parser url

	if (current_filename_ != file_name)
	{
		if (base_group_id_ > 0)
		{
			H5Gclose(base_group_id_);
			base_group_id_ = -1;
		}

		if (base_file_id_ > 0)
		{
			H5Fclose(base_file_id_);
			base_file_id_ = -1;
		}

	}

	if (base_file_id_ <= 0)
	{
		std::tie(current_filename_, base_file_id_) = open_file(file_name,
				is_append);
	}

	if (current_groupname_ != grp_name)
	{
		if (base_group_id_ > 0)
		{
			H5Gclose(base_group_id_);
			base_group_id_ = -1;
		}

	}

	if (base_group_id_ <= 0)
	{
		std::tie(current_groupname_, base_group_id_) = open_group(grp_name);
	}

	return pwd();
}

void DataStream::pimpl_s::close()
{

	if (base_group_id_ > 0)
	{
		H5Gclose(base_group_id_);
		base_group_id_ = -1;
	}

	if (base_file_id_ > 0)
	{
		H5Fclose(base_file_id_);
		base_file_id_ = -1;
	}

}

void DataStream::pimpl_s::flush_all()
{
	for (auto & item : cache_)
	{
		LOGGER << "\"" << flush_cache(item.first)
				<< "\" is flushed to hard disk!";
	}
}

void DataStream::pimpl_s::set_attribute(std::string const &url,
		DataType const &d_type, void const * buff)
{

	delete_attribute(url);

	std::string file_name, grp_path, obj_name, attr_name;

	std::tie(file_name, grp_path, obj_name, attr_name) = parser_url(url);

	hid_t g_id;

	std::tie(grp_path, g_id) = open_group(grp_path);

	hid_t o_id =
			(obj_name != "") ?
					H5Oopen(g_id, obj_name.c_str(), H5P_DEFAULT) : g_id;

	if (d_type.is_same<std::string>())
	{
		std::string const& s_str = *reinterpret_cast<std::string const*>(buff);

		hid_t m_type = H5Tcopy(H5T_C_S1);

		H5Tset_size(m_type, s_str.size());

		H5Tset_strpad(m_type, H5T_STR_NULLTERM);

		hid_t m_space = H5Screate(H5S_SCALAR);

		hid_t a_id = H5Acreate(o_id, attr_name.c_str(), m_type, m_space,
		H5P_DEFAULT, H5P_DEFAULT);

		H5Awrite(a_id, m_type, s_str.c_str());

		H5Tclose(m_type);

		H5Aclose(a_id);
	}
	else
	{
		hid_t m_type = create_h5_datatype(d_type);

		hid_t m_space = H5Screate(H5S_SCALAR);

		hid_t a_id = H5Acreate(o_id, attr_name.c_str(), m_type, m_space,
		H5P_DEFAULT, H5P_DEFAULT);

		H5Awrite(a_id, m_type, buff);

		if (H5Tcommitted(m_type) > 0)
			H5Tclose(m_type);

		H5Aclose(a_id);

		H5Sclose(m_space);
	}

	if (o_id != g_id)
		H5Oclose(o_id);
	if (g_id != base_group_id_)
		H5Gclose(g_id);
}

void DataStream::pimpl_s::get_attribute(std::string const &url,
		DataType const &d_type, void * buff)
{
	UNIMPLEMENT;
}
void DataStream::pimpl_s::delete_attribute(std::string const &url)
{
	std::string file_name, grp_name, obj_name, attr_name;

	std::tie(file_name, grp_name, obj_name, attr_name) = parser_url(url);

	if (obj_name != "")
	{
		hid_t g_id;
		std::tie(grp_name, g_id) = open_group(grp_name);

		if (H5Aexists_by_name(g_id, obj_name.c_str(), attr_name.c_str(),
		H5P_DEFAULT))
		{
			H5Adelete_by_name(g_id, obj_name.c_str(), attr_name.c_str(),
			H5P_DEFAULT);
		}
		if (g_id != base_group_id_)
			H5Gclose(g_id);
	}

}

/**
 *
 * @param url =<local path>/<obj name>.<attribute>
 * @return
 */
std::tuple<std::string, std::string, std::string, std::string> DataStream::pimpl_s::parser_url(
		std::string const & url_hint)
{
	std::string file_name(current_filename_), grp_name(current_groupname_),
			obj_name(""), attribute("");

	std::string url = url_hint;

	auto it = url.find(':');

	if (it != std::string::npos)
	{
		file_name = url.substr(0, it);
		url = url.substr(it + 1);
	}

	it = url.rfind('/');

	if (it != std::string::npos)
	{
		grp_name = url.substr(0, it + 1);
		url = url.substr(it + 1);
	}

	it = url.rfind('.');

	if (it != std::string::npos)
	{
		attribute = url.substr(it + 1);
		obj_name = url.substr(0, it);
	}
	else
	{
		obj_name = url;
	}

	return std::make_tuple(file_name, grp_name, obj_name, attribute);

}

std::string DataStream::pimpl_s::write(std::string const &url, void const* v,
		h5_dataset ds)
{
	if ((ds.flag & (SP_UNORDER)) == (SP_UNORDER))
	{
		return write_array(url, v, ds);
	}

	if ((ds.flag & SP_RECORD) == SP_RECORD)
	{
		convert_record_dataset(&ds);
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

std::string DataStream::pimpl_s::write(std::string const &url,
		DataSet const &ds, size_t flag)
{
	return write(url, ds.data.get(), create_h5_dataset(ds, flag));
}

hid_t DataStream::pimpl_s::create_h5_datatype(DataType const &d_type,
		bool is_compact_array)
{

	hid_t res;

	if (!d_type.is_compound())
	{

		hid_t ele_type;

		if (d_type.t_index_ == std::type_index(typeid(int)))
		{
			ele_type = H5T_NATIVE_INT;
		}
		else if (d_type.t_index_ == std::type_index(typeid(long)))
		{
			ele_type = H5T_NATIVE_LONG;
		}
		else if (d_type.t_index_ == std::type_index(typeid(unsigned long)))
		{
			ele_type = H5T_NATIVE_ULONG;
		}
		else if (d_type.t_index_ == std::type_index(typeid(float)))
		{
			ele_type = H5T_NATIVE_FLOAT;
		}
		else if (d_type.t_index_ == std::type_index(typeid(double)))
		{
			ele_type = H5T_NATIVE_DOUBLE;
		}
		else if (d_type.t_index_
				== std::type_index(typeid(std::complex<double>)))
		{
			ele_type = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
			H5Tinsert(ele_type, "r", 0, H5T_NATIVE_DOUBLE);
			H5Tinsert(ele_type, "i", sizeof(double), H5T_NATIVE_DOUBLE);

		}

		if (is_compact_array && d_type.ndims > 0)
		{
			hsize_t dims[d_type.ndims];
			std::copy(d_type.dimensions_, d_type.dimensions_ + d_type.ndims,
					dims);
			res = H5Tarray_create(ele_type, d_type.ndims, dims);

		}
		else
		{
			res = ele_type;
		}

	}
	else
	{

		res = H5Tcreate(H5T_COMPOUND, d_type.size_in_byte());

		for (auto const & item : d_type.data)
		{
			H5Tinsert(res, std::get<1>(item).c_str(), std::get<2>(item),
					create_h5_datatype(std::get<0>(item), true));
		}

	}

	return (res);
}

DataStream::pimpl_s::h5_dataset DataStream::pimpl_s::create_h5_dataset(

DataType const & datatype,

size_t ndims,

size_t const *p_global_begin,

size_t const *p_global_end,

size_t const *p_local_outer_begin,

size_t const *p_local_outer_end,

size_t const *p_local_inner_begin,

size_t const *p_local_inner_end,

size_t flag) const
{
	h5_dataset res;

	res.datatype = datatype;

	res.flag = flag;

	for (int i = 0; i < ndims; ++i)
	{
		auto g_begin = (p_global_begin == nullptr) ? 0 : p_global_begin[i];

		res.f_dims[i] =
				(p_global_end == nullptr) ? 1 : p_global_end[i] - g_begin;

		res.f_stride[i] = res.f_dims[i];

		res.f_offset[i] =
				(p_local_inner_begin == nullptr) ?
						0 : p_local_inner_begin[i] - g_begin;

		res.m_dims[i] =
				(p_local_outer_end == nullptr || p_local_outer_begin == nullptr) ?
						res.f_dims[i] :
						p_local_outer_end[i] - p_local_outer_begin[i];

		res.m_offset[i] =
				(p_local_inner_begin == nullptr
						|| p_local_outer_begin == nullptr) ?
						0 : p_local_inner_begin[i] - p_local_outer_begin[i];

		res.m_stride[i] = res.m_dims[i];

		res.count[i] = 1;

		res.block[i] =
				(p_local_inner_end == nullptr || p_local_inner_begin == nullptr) ?
						res.f_dims[i] :
						p_local_inner_end[i] - p_local_inner_begin[i];

	}

	if ((flag & SP_UNORDER) == SP_UNORDER)
	{
		std::tie(res.f_offset[0], res.f_dims[0]) = sync_global_location(
				res.f_dims[0]);

		res.f_stride[0] = res.f_dims[0];
	}

	if (datatype.ndims > 0)
	{
		for (int j = 0; j < datatype.ndims; ++j)
		{

			res.f_dims[ndims + j] = datatype.dimensions_[j];
			res.f_offset[ndims + j] = 0;
			res.f_stride[ndims + j] = res.f_dims[ndims + j];

			res.m_dims[ndims + j] = datatype.dimensions_[j];
			res.m_offset[ndims + j] = 0;
			res.m_stride[ndims + j] = res.m_dims[ndims + j];

			res.count[ndims + j] = 1;
			res.block[ndims + j] = datatype.dimensions_[j];

		}

		ndims += datatype.ndims;
	}

	res.ndims = ndims;

	if (properties["Enable Compact Storage"].as<bool>(false))
	{
		res.flag |= SP_APPEND;
	}

	if (properties["Force Record Storage"].as<bool>(false))
	{
		res.flag |= SP_RECORD;
	}
	if (properties["Force Write Cache"].as<bool>(false))
	{
		res.flag |= SP_CACHE;
	}
	return std::move(res);

}

DataStream::pimpl_s::h5_dataset DataStream::pimpl_s::create_h5_dataset(
		DataSet const & ds, size_t flag) const
{
	static constexpr size_t MAX_NUM_DIMS = DataSpace::MAX_NUM_DIMS;

	size_t global_begin[MAX_NUM_DIMS];

	size_t global_end[MAX_NUM_DIMS];

	size_t local_outer_begin[MAX_NUM_DIMS];

	size_t local_outer_end[MAX_NUM_DIMS];

	size_t local_inner_begin[MAX_NUM_DIMS];

	size_t local_inner_end[MAX_NUM_DIMS];

	size_t ndims = ds.dataspace.get_shape(global_begin, global_end,
			local_outer_begin, local_outer_end, local_inner_begin,
			local_inner_end);

	return std::move(
			create_h5_dataset(ds.datatype, ndims, global_begin, global_end,
					local_outer_begin, local_outer_end, local_inner_begin,
					local_inner_end, flag));

}
void DataStream::pimpl_s::convert_record_dataset(h5_dataset *pds) const
{
	for (int i = pds->ndims; i > 0; --i)
	{

		pds->f_dims[i] = pds->f_dims[i - 1];
		pds->f_offset[i] = pds->f_offset[i - 1];
		pds->f_stride[i] = pds->f_stride[i - 1];
		pds->m_dims[i] = pds->m_dims[i - 1];
		pds->m_offset[i] = pds->m_offset[i - 1];
		pds->m_stride[i] = pds->m_stride[i - 1];
		pds->count[i] = pds->count[i - 1];
		pds->block[i] = pds->block[i - 1];

	}

	pds->f_dims[0] = 1;
	pds->f_offset[0] = 0;
	pds->f_stride[0] = 1;

	pds->m_dims[0] = 1;
	pds->m_offset[0] = 0;
	pds->m_stride[0] = 1;

	pds->count[0] = 1;
	pds->block[0] = 1;

	++pds->ndims;

	pds->flag |= SP_APPEND;

}

std::string DataStream::pimpl_s::write_cache(std::string const & p_url,
		const void *v, h5_dataset const & ds)
{

	std::string filename, grp_name, dsname;

	std::tie(filename, grp_name, dsname, std::ignore) = parser_url(p_url);

	cd(filename, grp_name, ds.flag);

	std::string url = pwd() + dsname;

	if (cache_.find(url) == cache_.end())
	{
		size_t cache_memory_size = ds.datatype.ele_size_in_byte_;
		for (int i = 0; i < ds.ndims; ++i)
		{
			cache_memory_size *= ds.m_dims[i];
		}

		size_t cache_depth = properties["Max Cache Size"].as<size_t>(
				10 * 1024 * 1024UL) / cache_memory_size;

		if (cache_depth <= properties["Min Cache Number"].as<int>(5))
		{
			return write_array(url, v, ds);
		}
		else
		{

			mempool_.make_shared<ByteType>(cache_memory_size * cache_depth).swap(
					std::get<0>(cache_[url]));

			h5_dataset & item = std::get<1>(cache_[url]);

			item.datatype = ds.datatype;

			item.flag = ds.flag | SP_APPEND;

			item.ndims = ds.ndims;

			for (int i = 0; i < ds.ndims; ++i)
			{

				item.f_dims[i] = ds.f_dims[i];

				item.f_offset[i] = ds.f_offset[i];

				item.f_stride[i] = ds.f_stride[i];

				item.m_dims[i] = ds.m_dims[i];

				item.m_offset[i] = ds.m_offset[i];

				item.m_stride[i] = ds.m_stride[i];

				item.count[i] = ds.count[i];

				item.block[i] = ds.block[i];

			}
			item.count[0] = 0;
			item.m_dims[0] = item.m_stride[0] * cache_depth + item.m_offset[0];
			item.f_dims[0] = item.f_stride[0] * cache_depth + item.f_offset[0];

		}
	}
	auto & data = std::get<0>(cache_[url]);
	auto & item = std::get<1>(cache_[url]);

	size_t memory_size = ds.datatype.ele_size_in_byte_ * item.m_stride[0];

	for (int i = 1; i < item.ndims; ++i)
	{
		memory_size *= item.m_dims[i];
	}

	std::memcpy(
			reinterpret_cast<void*>(data.get() + item.count[0] * memory_size),
			v, memory_size);

	++item.count[0];

	if (item.count[0] * item.f_stride[0] + item.f_offset[0] >= item.m_dims[0])
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

	hsize_t t_f_shape = item.f_dims[0];
	hsize_t t_m_shape = item.m_dims[0];

	item.m_dims[0] = item.count[0] * item.m_stride[0] + item.m_offset[0];
	item.f_dims[0] = item.count[0] * item.f_stride[0] + item.f_offset[0];

	auto res = write_array(url, data.get(), item);

	item.m_dims[0] = t_f_shape;
	item.f_dims[0] = t_m_shape;

	item.count[0] = 0;

	return res;
}

std::string DataStream::pimpl_s::write_array(std::string const & url,
		const void *v, h5_dataset const &ds)
{
//	if (v == nullptr)
//	{
//		WARNING << "Data is empty! Can not write to " << p_url;
//		return "";
//	}

//todo (salmon) need optimize for empty space

	std::string filename, grp_name, dsname;

	std::tie(filename, grp_name, dsname, std::ignore) = parser_url(url);

	cd(filename, grp_name, ds.flag);

	if (dsname != "" && (ds.flag & SP_APPEND) != SP_APPEND)
	{
#ifdef USE_MPI
		if (GLOBAL_COMM.get_rank() == 0)
#endif
		{
			dsname =
					dsname
							+

							AutoIncrease(
									[&](std::string const & s )->bool
									{
										return H5Lexists(base_group_id_, (dsname + s ).c_str(), H5P_DEFAULT) > 0;
									}, 0, 4);
		}

		sync_string(&dsname);
	}

	hid_t m_type = create_h5_datatype(ds.datatype);

	hid_t file_space, mem_space;

	hid_t dset;

	if ((ds.flag & SP_APPEND) == 0)
	{

		file_space = H5Screate_simple(ds.ndims, ds.f_dims, nullptr);

		dset = H5Dcreate(base_group_id_, dsname.c_str(), m_type, file_space,
		H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

		H5_ERROR(H5Sclose(file_space));

		H5_ERROR(H5Fflush(base_group_id_, H5F_SCOPE_GLOBAL));

		file_space = H5Dget_space(dset);

		if (check_null_dataset(ds))
		{
			mem_space = H5S_ALL;
		}
		else
		{

			H5_ERROR(
					H5Sselect_hyperslab(file_space, H5S_SELECT_SET, ds.f_offset,
							ds.f_stride, ds.count, ds.block));

			mem_space = H5Screate_simple(ds.ndims, ds.m_dims, NULL);

			H5_ERROR(
					H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, ds.m_offset,
							ds.m_stride, ds.count, ds.block));
		}

	}
	else
	{
		if (H5Lexists(base_group_id_, dsname.c_str(), H5P_DEFAULT) == 0)
		{
			int f_ndims = ds.ndims;
			hsize_t current_dims[MAX_NDIMS_OF_ARRAY];
			hsize_t maximum_dims[MAX_NDIMS_OF_ARRAY];

			std::copy(ds.f_dims, ds.f_dims + ds.ndims, current_dims);

			std::copy(ds.f_dims, ds.f_dims + ds.ndims, maximum_dims);

			hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);

			H5_ERROR(H5Pset_chunk(dcpl_id, f_ndims, current_dims));

			maximum_dims[0] = H5S_UNLIMITED;

			current_dims[0] = 0;

			hid_t f_space = H5Screate_simple(f_ndims, current_dims,
					maximum_dims);

			hid_t t_dset = H5Dcreate(base_group_id_, dsname.c_str(), m_type,
					f_space, H5P_DEFAULT, dcpl_id,
					H5P_DEFAULT);

			H5_ERROR(H5Sclose(f_space));

			H5_ERROR(H5Dclose(t_dset));

			H5_ERROR(H5Pclose(dcpl_id));

		}

		dset = H5Dopen(base_group_id_, dsname.c_str(), H5P_DEFAULT);

		if (check_null_dataset(ds))
		{
			mem_space = H5S_ALL;
			file_space = H5S_ALL;
		}
		else
		{

			file_space = H5Dget_space(dset);

			int ndims = H5Sget_simple_extent_ndims(file_space);

			hsize_t f_shape[MAX_NDIMS_OF_ARRAY];
			hsize_t f_offset[MAX_NDIMS_OF_ARRAY];

			H5Sget_simple_extent_dims(file_space, f_shape, nullptr);

			H5Sclose(file_space);

			std::copy(ds.f_offset, ds.f_offset + ndims, f_offset);

			f_offset[0] += f_shape[0];

			f_shape[0] += ds.f_dims[0];

			H5Dset_extent(dset, f_shape);

			file_space = H5Dget_space(dset);

			H5_ERROR(
					H5Sselect_hyperslab(file_space, H5S_SELECT_SET, f_offset,
							ds.f_stride, ds.count, ds.block));

			mem_space = H5Screate_simple(ds.ndims, ds.m_dims, nullptr);

			H5_ERROR(
					H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, ds.m_offset,
							ds.m_stride, ds.count, ds.block));
		}
		//	CHECK(ndims);
		//	CHECK(f_shape[0]) << " " << f_shape[1];
		//	CHECK(ds.f_offset[0]) << " " << ds.f_offset[1];
		//	CHECK(ds.m_shape[0]) << " " << ds.m_shape[1];
		//	CHECK(ds.m_offset[0]) << " " << ds.m_offset[1];
		//	CHECK(ds.m_stride[0]) << " " << ds.m_stride[1];
		//	CHECK(ds.count[0]) << " " << ds.count[1];
		//	CHECK(ds.block[0]) << " " << ds.block[1];
	}
	if (!check_null_dataset(ds) && v != nullptr)
	{
// create property list for collective DataSet write.
#ifdef USE_MPI

		hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
		H5_ERROR(H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT));
		H5_ERROR(H5Dwrite(dset, m_type, mem_space, file_space, plist_id, v));
		H5_ERROR(H5Pclose(plist_id));

#else
		H5_ERROR(H5Dwrite(dset,m_type , mem_space, file_space, H5P_DEFAULT, v));
#endif
	}

	H5_ERROR(H5Dclose(dset));

	if (mem_space != H5S_ALL)
		H5_ERROR(H5Sclose(mem_space));

	if (file_space != H5S_ALL)
		H5_ERROR(H5Sclose(file_space));

	if (H5Tcommitted(m_type) > 0)
		H5Tclose(m_type);

	return pwd() + dsname;
}

//=====================================================================================
DataStream::DataStream() :
		pimpl_(new pimpl_s)
{
}
DataStream::~DataStream()
{
}
bool DataStream::is_valid() const
{
	return pimpl_->is_valid();
}
void DataStream::init(int argc, char** argv)
{
	pimpl_->init(argc, argv);
}
std::string DataStream::cd(std::string const & url, size_t flag)
{
	return pimpl_->cd(url, flag);
}
Properties & DataStream::properties()
{
	return pimpl_->properties;
}
Properties const& DataStream::properties() const
{
	return pimpl_->properties;
}
std::string DataStream::pwd() const
{
	return pimpl_->pwd();
}
void DataStream::close()
{
	pimpl_->flush_all();
	return pimpl_->close();
}

void DataStream::set_attribute(std::string const &url, DataType const &d_type,
		void const * buff)
{
	pimpl_->set_attribute(url, d_type, buff);
}

void DataStream::get_attribute(std::string const &url, DataType const & d_type,
		void* buff)
{
	pimpl_->get_attribute(url, d_type, buff);
}

void DataStream::delete_attribute(std::string const &url)
{
	pimpl_->delete_attribute(url);
}

std::string DataStream::write(std::string const &name, DataSet const &ds,
		size_t flag) const
{
	return pimpl_->write(name, ds, flag);
}
std::string DataStream::write(std::string const &name, void const *v,

DataType const & datatype,

size_t ndims_or_number,

size_t const *global_begin,

size_t const *global_end,

size_t const *local_outer_begin,

size_t const *local_outer_end,

size_t const *local_inner_begin,

size_t const *local_inner_end,

size_t flag

)
{
	return pimpl_->write(name, v,

	pimpl_->create_h5_dataset(datatype, ndims_or_number,

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
