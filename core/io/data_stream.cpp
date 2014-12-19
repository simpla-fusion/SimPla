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

#if !NO_MPI || USE_MPI
#   include "../parallel/parallel.h"
#   include "../parallel/mpi_comm.h"
#   include "../parallel/mpi_aux_functions.h"
#endif

#include "../utilities/utilities.h"
#include "../utilities/memory_pool.h"

#define H5_ERROR( _FUN_ ) if((_FUN_)<0){Logger(LOG_ERROR) <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:\n HDF5 Error:";H5Eprint(H5E_DEFAULT, stderr);LOGGER<<std::endl;}

namespace simpla
{

struct DataStream::pimpl_s
{
	hid_t base_file_id_;
	hid_t base_group_id_;

	std::string current_groupname_;
	std::string current_filename_;

	typedef nTuple<hsize_t, MAX_NDIMS_OF_ARRAY> dims_type;

	std::tuple<std::string, std::string, std::string, std::string> parser_url(
			std::string const & url);
	std::tuple<std::string, hid_t> open_group(std::string const & path);
	std::tuple<std::string, hid_t> open_file(std::string const & path,
			bool is_append = false);

	std::string pwd() const;

	hid_t create_h5_datatype(DataType const &, size_t flag = 0UL) const;

	hid_t create_h5_dataspace(DataSpace const &, size_t flag = 0UL) const;

	//	typedef std::tuple<std::shared_ptr<ByteType>, h5_dataset> CacheDataSet;
	//
	//	std::map<std::string, CacheDataSet> cache_;
//	/**
//	 *  Convert 'DataSet' to 'h5_dataset'
//	 *
//	 * @param ds
//	 * @param flag
//	 * @return h5_dataset
//	 *
//	 * 	   if global_begin ==nullptr and mpi is enable,
//	 *      f_count[0] = sum( global_end[0], for all process)
//	 *      f_begin[0] = sum( global_end[0], for process < this process)
//	 *
//	 */

};

DataStream::DataStream() :
		pimpl_(new pimpl_s)
{

	pimpl_->base_file_id_ = -1;
	pimpl_->base_group_id_ = -1;
	pimpl_->current_filename_ = "untitle.h5";
	pimpl_->current_groupname_ = "/";

	hid_t error_stack = H5Eget_current_stack();
	H5Eset_auto(error_stack, NULL, NULL);

//	properties["File Name"] = std::string("");
//
//	properties["Group Name"] = std::string("/");
//
//	properties["Suffix Width"] = 4;
//
//	properties["Light Data Limit"] = 20;
//
//	properties["Enable Compact Storage"] = false;
//
//	properties["Enable XDMF"] = false;
//
//	properties["Cache Depth"] = static_cast<int>(50);

}
DataStream::~DataStream()
{
	if (pimpl_ != nullptr)
	{
		close();
	}
}

bool DataStream::is_valid() const
{
	return pimpl_ != nullptr && pimpl_->base_file_id_ > 0;
}
std::string DataStream::pwd() const
{
	return pimpl_->pwd();
//		properties["File Name"].as<std::string>() + ":" + properties["Group Name"].as<std::string>();
}
void DataStream::init(int argc, char** argv)
{

	if (pimpl_ == nullptr)
		pimpl_ = new pimpl_s;

	bool show_help = false;

	parse_cmd_line(argc, argv,

	[&,this](std::string const & opt,std::string const & value)->int
	{
		if(opt=="o"||opt=="prefix")
		{
			std::tie(pimpl_->current_filename_,pimpl_->current_groupname_,
					std::ignore,std::ignore)
			=pimpl_->parser_url(value);

//			properties.set("File Name",value);
		}
//		else if(opt=="force-write-cache")
//		{
//			properties.set("Force Write Cache",true);
//		}
//		else if(opt=="cache-depth")
//		{
//			properties.set("Cache Depth",ToValue<size_t>(value));
//		}
		else if(opt=="h"||opt=="help" )
		{
			show_help=true;
			return TERMINATE;
		}
		return CONTINUE;
	}

	);

	if (show_help)
	{
		SHOW_OPTIONS("-o,--prefix <STRING>", "output file path");
	}
	else
	{
//		pimpl_->current_filename_ = properties["File_Name"].template as<
//				std::string>();

		pimpl_->current_groupname_ = "/";
	}

	VERBOSE << "DataSteream is initialized!" << pimpl_->current_filename_
			<< std::endl;

}
void bcast_string(std::string * filename_)
{

#if !NO_MPI || USE_MPI

	if (!GLOBAL_COMM.is_valid()) return;

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

std::tuple<bool, std::string> DataStream::cd(std::string const &url,
		size_t flag)
{
	std::string file_name = pimpl_->current_filename_;
	std::string grp_name = pimpl_->current_groupname_;
	std::string obj_name = "";

	if (url != "")
	{
		std::tie(file_name, grp_name, obj_name, std::ignore) =
				pimpl_->parser_url(url);
	}

	//TODO using regex parser url

	if (pimpl_->current_filename_ != file_name)
	{
		if (pimpl_->base_group_id_ > 0)
		{
			H5Gclose(pimpl_->base_group_id_);
			pimpl_->base_group_id_ = -1;
		}

		if (pimpl_->base_file_id_ > 0)
		{
			H5Fclose(pimpl_->base_file_id_);
			pimpl_->base_file_id_ = -1;
		}

	}

	if (pimpl_->base_file_id_ <= 0)
	{
		std::tie(pimpl_->current_filename_, pimpl_->base_file_id_) =
				pimpl_->open_file(file_name, flag);
	}

	if (pimpl_->current_groupname_ != grp_name)
	{
		if (pimpl_->base_group_id_ > 0)
		{
			H5Gclose(pimpl_->base_group_id_);
			pimpl_->base_group_id_ = -1;
		}

	}

	if (pimpl_->base_group_id_ <= 0)
	{
		std::tie(pimpl_->current_groupname_, pimpl_->base_group_id_) =
				pimpl_->open_group(grp_name);
	}

	if (obj_name != "" && ((flag & (SP_APPEND | SP_RECORD)) == 0UL))
	{
#if !NO_MPI || USE_MPI
		if (GLOBAL_COMM.get_rank() == 0)
#endif
		{
			obj_name = obj_name +

			AutoIncrease([&](std::string const & s )->bool
			{
				return H5Lexists(pimpl_->base_group_id_,
						(obj_name + s ).c_str(), H5P_DEFAULT) > 0;
			}, 0, 4);
		}

		bcast_string(&obj_name);
	}

	bool is_existed = false;

	if (obj_name != "")
		is_existed = H5Lexists(pimpl_->base_group_id_, obj_name.c_str(),
		H5P_DEFAULT) != 0;

	return std::make_tuple(is_existed, obj_name);
}

void DataStream::close()
{

	if (pimpl_ != nullptr)
	{
		if (pimpl_->base_group_id_ > 0)
		{
			H5Gclose(pimpl_->base_group_id_);
			pimpl_->base_group_id_ = -1;
		}

		if (pimpl_->base_file_id_ > 0)
		{
//			H5Fclose(pimpl_->base_file_id_);
//			pimpl_->base_file_id_ = -1;
		}
		delete pimpl_;
		pimpl_ = nullptr;
	}
	VERBOSE << "DataSteream is closed" << std::endl;
}

//void DataStream::flush_all()
//{
//	for (auto & item : pimpl_->cache_)
//	{
//		LOGGER << "\"" << pimpl_->flush_cache(item.first)
//				<< "\" is flushed to hard disk!";
//	}
//}

void DataStream::set_attribute(std::string const &url, DataType const &d_type,
		void const * buff)
{

	delete_attribute(url);

	std::string file_name, grp_path, obj_name, attr_name;

	std::tie(file_name, grp_path, obj_name, attr_name) = pimpl_->parser_url(
			url);

	hid_t g_id;

	std::tie(grp_path, g_id) = pimpl_->open_group(grp_path);

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
		hid_t m_type = pimpl_->create_h5_datatype(d_type);

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
	if (g_id != pimpl_->base_group_id_)
		H5Gclose(g_id);
}

void DataStream::get_attribute(std::string const &url, DataType const &d_type,
		void * buff)
{
	UNIMPLEMENTED;
}
void DataStream::delete_attribute(std::string const &url)
{
	std::string file_name, grp_name, obj_name, attr_name;

	std::tie(file_name, grp_name, obj_name, attr_name) = pimpl_->parser_url(
			url);

	if (obj_name != "")
	{
		hid_t g_id;
		std::tie(grp_name, g_id) = pimpl_->open_group(grp_name);

		if (H5Aexists_by_name(g_id, obj_name.c_str(), attr_name.c_str(),
		H5P_DEFAULT))
		{
			H5Adelete_by_name(g_id, obj_name.c_str(), attr_name.c_str(),
			H5P_DEFAULT);
		}
		if (g_id != pimpl_->base_group_id_)
			H5Gclose(g_id);
	}

}

bool DataStream::set_attribute(std::string const &url,
		Properties const & d_type)
{
	UNIMPLEMENTED;
	// TODO UNIMPLEMENTED
	return false;
}

Properties DataStream::get_attribute(std::string const &url)
{
	UNIMPLEMENTED;
	// TODO UNIMPLEMENTED

	return std::move(Properties());
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

std::string DataStream::pimpl_s::pwd() const
{
	return (current_filename_ + ":" + current_groupname_);
//		properties["File Name"].as<std::string>() + ":" + properties["Group Name"].as<std::string>();
}

std::tuple<std::string, hid_t> DataStream::pimpl_s::open_file(
		std::string const & fname, bool is_append)
{
	std::string filename = fname;

	if (filename == "")
		filename = current_filename_;

	if (!is_append)
	{
#if !NO_MPI || USE_MPI
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

	hid_t plist_id;

	H5_ERROR(plist_id= H5Pcreate(H5P_FILE_ACCESS));

#if !NO_MPI || USE_MPI
	if (GLOBAL_COMM.is_valid())
	{
		H5Pset_fapl_mpio(plist_id, GLOBAL_COMM.comm(), GLOBAL_COMM.info());
	}
#endif

	hid_t f_id;

	H5_ERROR(
			f_id = H5Fcreate( filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id));

	H5_ERROR(H5Pclose(plist_id));

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

hid_t DataStream::pimpl_s::create_h5_datatype(DataType const &d_type,
		size_t is_compact_array) const
{
	hid_t res = H5T_NO_CLASS;

	if (!d_type.is_compound())
	{

		if (d_type.is_same<int>())
		{
			res = H5T_NATIVE_INT;
		}
		else if (d_type.is_same<long>())
		{
			res = H5T_NATIVE_LONG;
		}
		else if (d_type.is_same<unsigned long>())
		{
			res = H5T_NATIVE_ULONG;
		}
		else if (d_type.is_same<float>())
		{
			res = H5T_NATIVE_FLOAT;
		}
		else if (d_type.is_same<double>())
		{
			res = H5T_NATIVE_DOUBLE;
		}
		else if (d_type.is_same<std::complex<double>>())
		{
			H5_ERROR(
					res = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>)));
			H5_ERROR(H5Tinsert(res, "r", 0, H5T_NATIVE_DOUBLE));
			H5_ERROR(H5Tinsert(res, "i", sizeof(double), H5T_NATIVE_DOUBLE));

		}

		if (d_type.rank() > 0)
		{
			hsize_t dims[d_type.rank()];

			for (int i = 0; i < d_type.rank(); ++i)
			{
				dims[i] = d_type.extent(i);
			}

			H5_ERROR(res = H5Tarray_create(res, d_type.rank(), dims ));

		}

	}
	else
	{

		H5_ERROR(res = H5Tcreate(H5T_COMPOUND, d_type.size_in_byte()));

		for (auto const & item : d_type.members())
		{
			H5_ERROR(
					H5Tinsert(res, std::get<1>(item).c_str(), std::get<2>(item),
							create_h5_datatype(std::get<0>(item), true)));
		}

	}

	if (res == H5T_NO_CLASS)
	{
		WARNING << "H5 datatype convert failed!" << std::endl;
	}
	return (res);
}

hid_t DataStream::pimpl_s::create_h5_dataspace(DataSpace const &d_space,
		size_t flag) const
{

	size_t ndims;

	dims_type dims;

	dims_type offset;
	dims_type count;
	dims_type stride;
	dims_type block;

	std::tie(ndims, dims, offset, count, stride, block) = d_space.shape();

	if ((flag & SP_RECORD) != 0UL)
	{
		dims[ndims] = 1;
		offset[ndims] = 0;
		count[ndims] = 1;
		stride[ndims] = 1;
		block[ndims] = 1;
		++ndims;
	}

	dims_type max_dims;
	max_dims = dims;

	if ((flag & (SP_APPEND | SP_RECORD)) != 0UL)
	{
		max_dims[ndims - 1] = H5S_UNLIMITED;
	}

	hid_t res = H5Screate_simple(ndims, &dims[0], &max_dims[0]);

	H5_ERROR(
			H5Sselect_hyperslab(res, H5S_SELECT_SET, &offset[0], &stride[0],
					&count[0], &block[0]));

	return res;

}

std::string DataStream::write(std::string const & url, DataSet const &ds,
		size_t flag)
{

	if (!ds.is_valid())
	{
		WARNING << "Invalid dataset!" << url << std::endl;
		return "Invalid dataset: " + pwd();
	}

	std::string dsname = "";

	bool is_existed = false;

	std::tie(is_existed, dsname) = this->cd(url, flag);

	hid_t m_type = pimpl_->create_h5_datatype(ds.datatype);

	hid_t m_space = pimpl_->create_h5_dataspace(ds.dataspace.local_space());

	hid_t f_space = pimpl_->create_h5_dataspace(ds.dataspace, flag);

	hid_t dset;

	if (!is_existed)
	{

		hid_t dcpl_id = H5P_DEFAULT;

		if ((flag & (SP_APPEND | SP_RECORD)) != 0)
		{
			pimpl_s::dims_type current_dims;

			int f_ndims = H5Sget_simple_extent_ndims(f_space);

			H5_ERROR(
					H5Sget_simple_extent_dims(f_space, &current_dims[0],
							nullptr));

			H5_ERROR(dcpl_id = H5Pcreate(H5P_DATASET_CREATE));

			H5_ERROR(H5Pset_chunk(dcpl_id, f_ndims, &current_dims[0]));
		}

		H5_ERROR(
				dset = H5Dcreate(pimpl_->base_group_id_, dsname.c_str(), m_type, f_space, H5P_DEFAULT, dcpl_id, H5P_DEFAULT));

		if (dcpl_id != H5P_DEFAULT)
		{
			H5_ERROR(H5Pclose(dcpl_id));
		}
		H5_ERROR(H5Fflush(pimpl_->base_group_id_, H5F_SCOPE_GLOBAL));
	}
	else
	{

		H5_ERROR(
				dset = H5Dopen(pimpl_->base_group_id_, dsname.c_str(), H5P_DEFAULT));

		pimpl_s::dims_type current_dimensions;

		hid_t current_f_space;

		H5_ERROR(current_f_space = H5Dget_space(dset));

		int current_ndims = H5Sget_simple_extent_dims(current_f_space,
				&current_dimensions[0], nullptr);

		H5_ERROR(H5Sclose(current_f_space));

		pimpl_s::dims_type new_f_dimensions;
		pimpl_s::dims_type new_f_max_dimensions;
		pimpl_s::dims_type new_f_offset;
		pimpl_s::dims_type new_f_end;
		int new_f_ndims = H5Sget_simple_extent_dims(f_space,
				&new_f_dimensions[0], &new_f_max_dimensions[0]);

		H5_ERROR(H5Sget_select_bounds(f_space, &new_f_offset[0], &new_f_end[0]));

		ASSERT(current_ndims == current_ndims);
		ASSERT(new_f_max_dimensions[new_f_ndims-1]==H5S_UNLIMITED);

		new_f_dimensions[new_f_ndims - 1] +=
				current_dimensions[new_f_ndims - 1];

		new_f_offset[new_f_ndims - 1] += current_dimensions[new_f_ndims - 1];

		H5_ERROR(H5Dset_extent(dset, &new_f_dimensions[0]));

		H5_ERROR(
				H5Sset_extent_simple(f_space, new_f_ndims, &new_f_dimensions[0],
						&new_f_max_dimensions[0]));

		nTuple<hssize_t, MAX_NDIMS_OF_ARRAY> new_f_offset2;

		new_f_offset2 = new_f_offset;

		H5_ERROR(H5Soffset_simple(f_space, &new_f_offset2[0]));

	}

// create property list for collective DataSet write.
#if !NO_MPI || USE_MPI
	if (GLOBAL_COMM.is_valid())
	{
		hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
		H5_ERROR(H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT));
		H5_ERROR(H5Dwrite(dset, m_type, m_space, f_space, plist_id, ds.data.get()));
		H5_ERROR(H5Pclose(plist_id));
	}
	else
#endif
	{
		H5_ERROR( H5Dwrite(dset, m_type , m_space, f_space,
				H5P_DEFAULT, ds.data.get()));
	}



	H5_ERROR(H5Dclose(dset));

	if (m_space != H5S_ALL)
		H5_ERROR(H5Sclose(m_space));

	if (f_space != H5S_ALL)
		H5_ERROR(H5Sclose(f_space));

	if (H5Tcommitted(m_type) > 0)
	{
		H5_ERROR(H5Tclose(m_type));
	}

	return pwd() + dsname;
}
std::string DataStream::read(std::string const & url, DataSet*ds, size_t flag)
{
	UNIMPLEMENTED;
	return "UNIMPLEMENTED";
}
//hid_t DataStream::pimpl_s::create_h5_dataset(DataSet const & ds,
//		size_t flag) const
//{
//
//	h5_dataset res;
//
//	res.data = ds.data;
//
//	res.datatype = ds.datatype;
//
//	res.flag = flag;
//
//	res.ndims = ds.dataspace.num_of_dims();
//
//	std::tie(res.f_start, res.f_count) = ds.dataspace.global_shape();
//
//	std::tie(res.m_start, res.m_count) = ds.dataspace.local_shape();
//
//	std::tie(res.start, res.count, res.stride, res.block) =
//			ds.dataspace.shape();
//
//	if ((flag & SP_UNORDER) == SP_UNORDER)
//	{
//		std::tie(res.f_start[0], res.f_count[0]) = sync_global_location(
//				res.f_count[0]);
//
//		res.f_stride[0] = res.f_count[0];
//	}
//
//	if (ds.datatype.ndims > 0)
//	{
//		for (int j = 0; j < ds.datatype.ndims; ++j)
//		{
//
//			res.f_count[res.ndims + j] = ds.datatype.dimensions_[j];
//			res.f_start[res.ndims + j] = 0;
//			res.f_stride[res.ndims + j] = res.f_count[res.ndims + j];
//
//			res.m_count[res.ndims + j] = ds.datatype.dimensions_[j];
//			res.m_start[res.ndims + j] = 0;
//			res.m_stride[res.ndims + j] = res.m_count[res.ndims + j];
//
//			res.count[res.ndims + j] = 1;
//			res.block[res.ndims + j] = ds.datatype.dimensions_[j];
//
//		}
//
//		res.ndims += ds.datatype.ndims;
//	}
//
//	if (properties["Enable Compact Storage"].as<bool>(false))
//	{
//		res.flag |= SP_APPEND;
//	}
//
//	if (properties["Force Record Storage"].as<bool>(false))
//	{
//		res.flag |= SP_RECORD;
//	}
//	if (properties["Force Write Cache"].as<bool>(false))
//	{
//		res.flag |= SP_CACHE;
//	}
//	return std::move(res);
//
//}

//std::string DataStream::pimpl_s::write(std::string const &url, h5_dataset ds)
//{
//	if ((ds.flag & (SP_UNORDER)) == (SP_UNORDER))
//	{
//		return write_array(url, ds);
//	}
//
//	if ((ds.flag & SP_RECORD) == SP_RECORD)
//	{
//		convert_record_dataset(&ds);
//	}
//
//	if ((ds.flag & SP_CACHE) == SP_CACHE)
//	{
//		return write_cache(url, ds);
//	}
//	else
//	{
//		return write_array(url, ds);
//	}
//
//}

//void DataStream::pimpl_s::convert_record_dataset(h5_dataset *pds) const
//{
//	for (int i = pds->ndims; i > 0; --i)
//	{
//
//		pds->f_count[i] = pds->f_count[i - 1];
//		pds->f_start[i] = pds->f_start[i - 1];
//		pds->f_stride[i] = pds->f_stride[i - 1];
//		pds->m_count[i] = pds->m_count[i - 1];
//		pds->m_start[i] = pds->m_start[i - 1];
//		pds->m_stride[i] = pds->m_stride[i - 1];
//		pds->count[i] = pds->count[i - 1];
//		pds->block[i] = pds->block[i - 1];
//
//	}
//
//	pds->f_count[0] = 1;
//	pds->f_start[0] = 0;
//	pds->f_stride[0] = 1;
//
//	pds->m_count[0] = 1;
//	pds->m_start[0] = 0;
//	pds->m_stride[0] = 1;
//
//	pds->count[0] = 1;
//	pds->block[0] = 1;
//
//	++pds->ndims;
//
//	pds->flag |= SP_APPEND;
//
//}

//std::string DataStream::pimpl_s::write_cache(std::string const & p_url,
//		h5_dataset const & ds)
//{
//
//	std::string filename, grp_name, dsname;
//
//	std::tie(filename, grp_name, dsname, std::ignore) = parser_url(p_url);
//
//	cd(filename, grp_name, ds.flag);
//
//	std::string url = pwd() + dsname;
//
//	if (cache_.find(url) == cache_.end())
//	{
//		size_t cache_memory_size = ds.datatype.ele_size_in_byte_;
//		for (int i = 0; i < ds.ndims; ++i)
//		{
//			cache_memory_size *= ds.m_count[i];
//		}
//
//		size_t cache_depth = properties["Max Cache Size"].as<size_t>(
//				10 * 1024 * 1024UL) / cache_memory_size;
//
//		if (cache_depth <= properties["Min Cache Number"].as<int>(5))
//		{
//			return write_array(url, ds);
//		}
//		else
//		{
//			sp_make_shared_array<ByteType>(cache_memory_size * cache_depth).swap(
//					std::get<0>(cache_[url]));
//
//			h5_dataset & item = std::get<1>(cache_[url]);
//
//			item = ds;
//
//			item.flag |= SP_APPEND;
//
//			item.ndims = ds.ndims;
//
//			item.count[0] = 0;
//			item.m_count[0] = item.m_stride[0] * cache_depth + item.m_start[0];
//			item.f_count[0] = item.f_stride[0] * cache_depth + item.f_start[0];
//
//		}
//	}
//	auto & data = std::get<0>(cache_[url]);
//	auto & item = std::get<1>(cache_[url]);
//
//	size_t memory_size = ds.datatype.ele_size_in_byte_ * item.m_stride[0];
//
//	for (int i = 1; i < item.ndims; ++i)
//	{
//		memory_size *= item.m_count[i];
//	}
//
//	std::memcpy(
//			reinterpret_cast<void*>(data.get() + item.count[0] * memory_size),
//			ds.data.get(), memory_size);
//
//	++item.count[0];
//
//	if (item.count[0] * item.f_stride[0] + item.f_start[0] >= item.m_count[0])
//	{
//		return flush_cache(url);
//	}
//	else
//	{
//		return "\"" + url + "\" is write to cache";
//	}
//
//}
//std::string DataStream::pimpl_s::flush_cache(std::string const & url)
//{
//
//	if (cache_.find(url) == cache_.end())
//	{
//		return url + " is not found !";
//	}
//
//	auto & data = std::get<0>(cache_[url]);
//	auto & item = std::get<1>(cache_[url]);
//
//	hsize_t t_f_shape = item.f_count[0];
//	hsize_t t_m_shape = item.m_count[0];
//
//	item.m_count[0] = item.count[0] * item.m_stride[0] + item.m_start[0];
//	item.f_count[0] = item.count[0] * item.f_stride[0] + item.f_start[0];
//
//	auto res = write_array(url, item);
//
//	item.m_count[0] = t_f_shape;
//	item.f_count[0] = t_m_shape;
//
//	item.count[0] = 0;
//
//	return res;
//}

//=====================================================================================

}
// namespace simpla
