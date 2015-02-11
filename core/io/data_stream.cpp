
/**
 * @file data_stream.cpp
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

#include "../dataset/dataset.h"

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

	hid_t convert_data_type_sp_to_h5(DataType const &, size_t flag = 0UL) const;

	hid_t convert_data_space_sp_to_h5(DataSpace const &,
			size_t flag = 0UL) const;

	DataType convert_data_type_h5_to_sp(hid_t) const;

	DataSpace convert_data_space_h5_to_sp(hid_t) const;

	void set_attribute(hid_t loc_id, std::string const & name, Any const & v);
	void set_attribute(hid_t loc_id, Properties const & v);

	Any get_attribute(hid_t loc_id, std::string const & name) const;
	Properties get_attribute(hid_t loc_id, int idx = -1) const;

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
		}
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
		pimpl_->current_groupname_ = "/";
	}

	VERBOSE << "DataSteream is initialized!" << pimpl_->current_filename_
			<< std::endl;

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
			H5_ERROR(H5Gclose(pimpl_->base_group_id_));
			pimpl_->base_group_id_ = -1;
		}

		if (pimpl_->base_file_id_ > 0)
		{
			H5_ERROR(H5Fclose(pimpl_->base_file_id_));
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
			H5_ERROR(H5Gclose(pimpl_->base_group_id_));
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
		if (GLOBAL_COMM.get_rank() == 0)
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
			H5_ERROR(H5Gclose(pimpl_->base_group_id_));
			pimpl_->base_group_id_ = -1;
		}

		if (pimpl_->base_file_id_ > 0)
		{
			H5_ERROR(H5Fclose(pimpl_->base_file_id_));
			pimpl_->base_file_id_ = -1;
		}
		delete pimpl_;
		pimpl_ = nullptr;
	}
	VERBOSE << "DataSteream is closed" << std::endl;
}

void DataStream::set_attribute(std::string const &url, Any const & any_v)
{

	delete_attribute(url);

	DataType dtype = any_v.datatype();

	void const * v = any_v.data();

	std::string file_name, grp_path, obj_name, attr_name;

	std::tie(file_name, grp_path, obj_name, attr_name) = pimpl_->parser_url(
			url);

	hid_t g_id, o_id;

	std::tie(grp_path, g_id) = pimpl_->open_group(grp_path);

	if (o_id != g_id)
		H5Oclose(o_id);

	if (g_id != pimpl_->base_group_id_)
		H5Gclose(g_id);

}
void DataStream::pimpl_s::set_attribute(hid_t loc_id, std::string const &name,
		Any const & any_v)
{

	if (any_v.is_same<std::string>())
	{
		std::string const& s_str = any_v.as<std::string>();

		hid_t m_type = H5Tcopy(H5T_C_S1);

		H5Tset_size(m_type, s_str.size());

		H5Tset_strpad(m_type, H5T_STR_NULLTERM);

		hid_t m_space = H5Screate(H5S_SCALAR);

		hid_t a_id = H5Acreate(loc_id, name.c_str(), m_type, m_space,
		H5P_DEFAULT, H5P_DEFAULT);

		H5Awrite(a_id, m_type, s_str.c_str());

		H5Tclose(m_type);

		H5Aclose(a_id);
	}
	else
	{
		hid_t m_type = convert_data_type_sp_to_h5(any_v.datatype());

		hid_t m_space = H5Screate(H5S_SCALAR);

		hid_t a_id = H5Acreate(loc_id, name.c_str(), m_type, m_space,
		H5P_DEFAULT, H5P_DEFAULT);

		H5Awrite(a_id, m_type, any_v.data());

		if (H5Tcommitted(m_type) > 0)
			H5Tclose(m_type);

		H5Aclose(a_id);

		H5Sclose(m_space);
	}

}
void DataStream::pimpl_s::set_attribute(hid_t loc_id, Properties const & prop)
{
	for (auto const & item : prop)
	{
		set_attribute(loc_id, item.first, item.second);
	}

}
Any DataStream::pimpl_s::get_attribute(hid_t loc_id,
		std::string const &name) const
{

	UNIMPLEMENTED;
	return std::move(Any());
}

Properties DataStream::pimpl_s::get_attribute(hid_t loc_id, int idx) const
{
	UNIMPLEMENTED;
	Properties res;

	return std::move(res);
}

Any DataStream::get_attribute(std::string const &url) const
{
	UNIMPLEMENTED;
	return std::move(Any());
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

	if (!is_append && GLOBAL_COMM.get_rank() == 0)
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

	bcast_string(&filename);

	hid_t plist_id;

	H5_ERROR(plist_id= H5Pcreate(H5P_FILE_ACCESS));

	if (GLOBAL_COMM.is_valid())
	{
		H5Pset_fapl_mpio(plist_id, GLOBAL_COMM.comm(), GLOBAL_COMM.info());
	}

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

hid_t DataStream::pimpl_s::convert_data_type_sp_to_h5(DataType const &d_type,
		size_t is_compact_array) const
{
	hid_t res = H5T_NO_CLASS;

	if (!d_type.is_valid())
		RUNTIME_ERROR("illegal data type");

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
		else
		{
			RUNTIME_ERROR("Unknown data type:" + d_type.name());
		}

		if (d_type.is_array())
		{
			hsize_t dims[d_type.rank()];

			for (int i = 0; i < d_type.rank(); ++i)
			{
				dims[i] = d_type.extent(i);
			}
			hid_t res2 = res;

			H5_ERROR(res2 = H5Tarray_create(res, d_type.rank(), dims ));

			if (H5Tcommitted(res) > 0)
				H5_ERROR(H5Tclose(res));

			res = res2;
		}
	}
	else
	{
		H5_ERROR(res = H5Tcreate(H5T_COMPOUND, d_type.size_in_byte()));

		for (auto const & item : d_type.members())
		{
			hid_t t_member = convert_data_type_sp_to_h5(std::get<0>(item),
			true);
			H5_ERROR(
					H5Tinsert(res, std::get<1>(item).c_str(), std::get<2>(item),
							t_member));
			if (H5Tcommitted(t_member) > 0)
				H5_ERROR(H5Tclose(t_member));
		}
	}

	if (res == H5T_NO_CLASS)
	{
		WARNING << "sp.DataType convert to H5.datatype failed!" << std::endl;
		throw std::bad_cast();
	}
	return (res);
}

DataType DataStream::pimpl_s::convert_data_type_h5_to_sp(hid_t t_id) const
{

	bool bad_cast_error = true;

	DataType dtype;

	H5T_class_t type_class = H5Tget_class(t_id);

	if (type_class == H5T_NO_CLASS)
	{
		bad_cast_error = true;
	}
	else if (type_class == H5T_OPAQUE)
	{
		DataType(std::type_index(typeid(void)), H5Tget_size(t_id)).swap(dtype);
	}
	else if (type_class == H5T_COMPOUND)
	{
		for (int i = 0, num = H5Tget_nmembers(t_id); i < num; ++i)
		{
			dtype.push_back(
					convert_data_type_h5_to_sp(H5Tget_member_type(t_id, i)),
					std::string(H5Tget_member_name(t_id, i)),
					H5Tget_member_offset(t_id, i));
		}

	}
	else if (type_class == H5T_INTEGER || type_class == H5T_FLOAT
			|| type_class == H5T_ARRAY)
	{
		hid_t atomic_id = H5Tget_native_type(type_class, H5T_DIR_ASCEND);

		UNIMPLEMENTED;

//		switch (atomic_id)
//		{
//		case H5T_NATIVE_CHAR:
//			make_datatype<char>().swap(dtype);
//			break;
//		case H5T_NATIVE_SHORT:
//			make_datatype<short>().swap(dtype);
//			break;
//		case H5T_NATIVE_INT:
//			make_datatype<int>().swap(dtype);
//			break;
//		case H5T_NATIVE_LONG:
//			make_datatype<long>().swap(dtype);
//			break;
//		case H5T_NATIVE_LLONG:
//			make_datatype<long long>().swap(dtype);
//			break;
//		case H5T_NATIVE_UCHAR:
//			make_datatype<unsigned char>().swap(dtype);
//			break;
//		case H5T_NATIVE_USHORT:
//			make_datatype<unsigned short>().swap(dtype);
//			break;
//		case H5T_NATIVE_UINT:
//			make_datatype<unsigned int>().swap(dtype);
//			break;
//		case H5T_NATIVE_ULONG:
//			make_datatype<unsigned long>().swap(dtype);
//			break;
//		case H5T_NATIVE_ULLONG:
//			make_datatype<unsigned long long>().swap(dtype);
//			break;
//		case H5T_NATIVE_FLOAT:
//			make_datatype<float>().swap(dtype);
//			break;
//		case H5T_NATIVE_DOUBLE:
//			make_datatype<double>().swap(dtype);
//			break;
//		case H5T_NATIVE_LDOUBLE:
//			make_datatype<long double>().swap(dtype);
//			break;
//		default:
//			bad_cast_error = true;
//
//		}

		H5_ERROR(H5Tclose(atomic_id));
	}
	else if (type_class == H5T_TIME)
	{
		UNIMPLEMENTED;
	}
	else if (type_class == H5T_STRING)
	{
		UNIMPLEMENTED;
	}
	else if (type_class == H5T_BITFIELD)
	{
		UNIMPLEMENTED;
	}
	else if (type_class == H5T_REFERENCE)
	{
		UNIMPLEMENTED;
	}
	else if (type_class == H5T_ENUM)
	{
		UNIMPLEMENTED;
	}
	else if (type_class == H5T_VLEN)
	{
		UNIMPLEMENTED;
	}

	if (type_class == H5T_ARRAY)
	{
		int rank = H5Tget_array_ndims(t_id);
		hsize_t dims[rank];
		size_t dims2[rank];
		for (int i = 0; i < rank; ++i)
		{
			dims2[i] = dims[i];
		}
		H5_ERROR(H5Tget_array_dims(t_id,dims));

		dtype.extent(rank, dims2);
	}
	if (bad_cast_error)
	{
		Logger(LOG_ERROR) << "H5 datatype convert to sp.DataType failed!"
				<< std::endl;
		throw std::bad_cast();
	}

	return std::move(dtype);

}

hid_t DataStream::pimpl_s::convert_data_space_sp_to_h5(DataSpace const &d_space,
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
DataSpace DataStream::pimpl_s::convert_data_space_h5_to_sp(hid_t) const
{
	return DataSpace();
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

	hid_t m_type = pimpl_->convert_data_type_sp_to_h5(ds.datatype);

	hid_t m_space = pimpl_->convert_data_space_sp_to_h5(
			ds.dataspace.local_space());

	hid_t f_space = pimpl_->convert_data_space_sp_to_h5(ds.dataspace, flag);

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

//		ASSERT(current_ndims == current_ndims);
//		ASSERT(new_f_max_dimensions[new_f_ndims-1]==H5S_UNLIMITED);

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

	pimpl_->set_attribute(dset, ds.properties);

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
