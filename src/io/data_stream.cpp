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
#include <mpi.h>
}

#include "hdf5_datatype.h"
#include "data_stream.h"
#include "../parallel/parallel.h"
#include "../parallel/message_comm.h"
#include "../parallel/mpi_datatype.h"
#include "../utilities/properties.h"

#define H5_ERROR( _FUN_ ) if((_FUN_)<0){ H5Eprint(H5E_DEFAULT, stderr);}

namespace simpla
{

struct DataStream::pimpl_s
{
	hid_t file_;
	hid_t group_;

//	std::string prefix_;
//	std::string filename_;
//	std::string grpname_;
//	int suffix_width_;
//	size_t LIGHT_DATA_LIMIT_;
//	bool enable_compact_storable_;
//	bool enable_xdmf_;

	Properties prop_;

public:

	pimpl_s();
	~pimpl_s();

	bool is_ready()
	{
		return file_ > 0;
	}
	void set_property(std::string const & name, Any const&v)
	{
		prop_[name] = v;
	}
	Any const & get_property_any(std::string const &name) const
	{
		return prop_[name].template as<Any>();
	}

	void init(int argc = 0, char** argv = nullptr);
	void open_group(std::string const & gname);
	void open_file(std::string const &fname = "unnamed");
	void close_group();
	void close_file();
	void close();

	inline std::string GetCurrentPath() const
	{
		return prop_["File Name"].template as<std::string>() + ":" + prop_["Group Name"].template as<std::string>();
	}

	std::string write(std::string const &name, void const *v,

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

	std::string write2(std::string const &name, void const *v, DataType const & data_desc, size_t count) const;
};

DataStream::pimpl_s::pimpl_s()
		: file_(-1), group_(-1)
{
	hid_t error_stack = H5Eget_current_stack();
	H5Eset_auto(error_stack, NULL, NULL);

	prop_["Prefix"] = std::string("simpla_unnamed");

	prop_["File Name"] = std::string("unnamed");

	prop_["Group Name"] = std::string("");

	prop_["Suffix Width"] = 4;

	prop_["Light Data Limit"] = 20;

	prop_["Enable Compact Storage"] = false;

	prop_["Enable XDMF"] = false;

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

void DataStream::pimpl_s::open_group(std::string const & gname)
{
	if (gname == "")
		return;

	hid_t h5fg = file_;

	close_group();

	std::string & grpname_ = prop_["Group Name"].template as<std::string>();

	if (gname[0] == '/')
	{
		grpname_ = gname;
	}
	else
	{
		grpname_ += gname;
		if (group_ > 0)
			h5fg = group_;
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
		RUNTIME_ERROR("Can not open group " + grpname_ + " in file " + prop_["Prefix"].template as<std::string>());
	}

}

void sync_string(std::string * filename_)
{

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

}

void DataStream::pimpl_s::open_file(std::string const &fname)
{

	close_file();

	GLOBAL_COMM.Barrier();

	std::string filename_ = prop_["File Name"].template as<std::string>();
	std::string & prefix = prop_["Prefix"].template as<std::string>();

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
	// sync filename and open file
	if (GLOBAL_COMM.is_ready())
	{
		sync_string(& filename_);

		hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

		H5Pset_fapl_mpio(plist_id, GLOBAL_COMM.comm(), GLOBAL_COMM.info());

		H5_ERROR(file_ = H5Fcreate( filename_.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id));

		H5Pclose(plist_id);

	}
	else
	{
		H5_ERROR(file_ = H5Fcreate( filename_.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT));
	}
	GLOBAL_COMM.Barrier();

	if (file_ < 0)
	{
		RUNTIME_ERROR("create HDF5 file " + filename_ + " failed!");
	}

	prop_["File Name"].template as<std::string>() = filename_;
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
std::string DataStream::pimpl_s::write(std::string const & name, void const *v,

DataType const & data_desc,

size_t ndims_or_number,

size_t const *p_global_begin,

size_t const *p_global_end,

size_t const *p_local_outer_begin,

size_t const *p_local_outer_end,

size_t const *p_local_inner_begin,

size_t const *p_local_inner_end,

unsigned int flag

) const
{

	assert((flag&SP_FAST_FIRST) == 0UL);

	int rank = 1;

	if (p_global_end == nullptr)
	{
		return write2(name, v, data_desc, ndims_or_number);
	}
	else
	{
		rank = ndims_or_number;
	}
	/// @todo add support for FASF_FIRST array
//
//	CHECK(rank);
//	CHECK(p_global_begin[0]) << " " << p_global_end[0];
//	CHECK(p_global_begin[1]) << " " << p_global_end[1];
//
//	CHECK(p_local_outer_begin[0]) << " " << p_local_outer_end[0];
//	CHECK(p_local_outer_begin[1]) << " " << p_local_outer_end[1];
//
//	CHECK(p_local_inner_begin[0]) << " " << p_local_inner_end[0];
//	CHECK(p_local_inner_begin[1]) << " " << p_local_inner_end[1];

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
	hsize_t g_begin[rank + data_desc.NDIMS + 1];
	hsize_t g_shape[rank + data_desc.NDIMS + 1];
	hsize_t f_begin[rank + data_desc.NDIMS + 1];
	hsize_t m_shape[rank + data_desc.NDIMS + 1];
	hsize_t m_begin[rank + data_desc.NDIMS + 1];
	hsize_t m_count[rank + data_desc.NDIMS + 1];

	for (int i = 0; i < rank; ++i)
	{
		g_begin[i] = (p_global_begin == nullptr) ? 0 : p_global_begin[i];

		g_shape[i] = (p_global_end == nullptr) ? 1 : p_global_end[i] - g_begin[i];

		f_begin[i] = (p_local_inner_begin == nullptr) ? 0 : p_local_inner_begin[i] - g_begin[i];

		m_shape[i] =
		        (p_local_outer_end == nullptr || p_local_outer_begin == nullptr) ?
		                g_shape[i] : p_local_outer_end[i] - p_local_outer_begin[i];

		m_begin[i] =
		        (p_local_inner_begin == nullptr || p_local_outer_begin == nullptr) ?
		                0 : p_local_inner_begin[i] - p_local_outer_begin[i];

		m_count[i] =
		        (p_local_inner_end == nullptr || p_local_inner_begin == nullptr) ?
		                g_shape[i] : p_local_inner_end[i] - p_local_inner_begin[i];
	}

	if (data_desc.NDIMS > 0)
	{
		for (int j = 0; j < data_desc.NDIMS; ++j)
		{

			g_shape[rank + j] = data_desc.dimensions_[j];
			f_begin[rank + j] = 0;
			m_shape[rank + j] = data_desc.dimensions_[j];
			m_begin[rank + j] = 0;
			m_count[rank + j] = data_desc.dimensions_[j];

			++rank;
		}
	}

//	CHECK(name);
//	CHECK(rank);
//	CHECK(g_shape[0]) << " " << g_shape[1] << " " << g_shape[2] << " " << g_shape[3];
//	CHECK(f_begin[0]) << " " << f_begin[1] << " " << f_begin[2] << " " << f_begin[3];
//	CHECK(m_shape[0]) << " " << m_shape[1] << " " << m_shape[2] << " " << m_shape[3];
//	CHECK(m_begin[0]) << " " << m_begin[1] << " " << m_begin[2] << " " << m_begin[3];
//	CHECK(m_count[0]) << " " << m_count[1] << " " << m_count[2] << " " << m_count[3];

	hid_t m_type = GLOBAL_HDF5_DATA_TYPE_FACTORY.create(data_desc.t_info_);

	hid_t dset;

	hid_t file_space, mem_space;

	if (!prop_["Enable Compact Storage"].template as<bool>())
	{
		if (GLOBAL_COMM.get_rank()==0)
		{
			dsname = dsname +

			AutoIncrease([&](std::string const & s )->bool
			{
				return H5Lexists(group_, (dsname + s ).c_str(), H5P_DEFAULT) > 0;
			}, 0, 4);
		}

		sync_string(&dsname);

		file_space = H5Screate_simple(rank, g_shape, nullptr);

		dset = H5Dcreate(group_, dsname.c_str(), m_type, file_space,

		H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

		H5_ERROR(H5Sclose(file_space));

		H5_ERROR(H5Fflush(group_, H5F_SCOPE_GLOBAL));

		file_space = H5Dget_space(dset);

		H5_ERROR(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, f_begin, NULL, m_count, NULL));

		mem_space = H5Screate_simple(rank, m_shape, NULL);

		H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, m_begin, NULL, m_count, NULL);

	}
	else if(!(flag&SP_APPEND)) // add new record, extent the 'rank' dimension; file data has rank+1 dimension
	{

		g_shape[rank] = 1;
		f_begin[rank] = 0;
		m_shape[rank] = 1;
		m_begin[rank] = 0;
		m_count[rank] = 0;

		if (H5Lexists(group_, dsname.c_str(), H5P_DEFAULT) == 0)
		{
			hsize_t max_dims[rank + 1];

			std::copy(g_shape, g_shape + rank, max_dims);

			max_dims[rank] = H5S_UNLIMITED;

			hid_t space = H5Screate_simple(rank + 1, g_shape, max_dims);

			hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);

			H5_ERROR(H5Pset_chunk(dcpl_id, rank + 1, g_shape));

			dset = H5Dcreate(group_, dsname.c_str(), m_type, space, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

			H5_ERROR(H5Sclose(space));

			H5_ERROR(H5Pclose(dcpl_id));

		}
		else
		{

			dset = H5Dopen(group_, dsname.c_str(), H5P_DEFAULT);

			file_space = H5Dget_space(dset);

			H5Sget_simple_extent_dims(file_space, g_shape, nullptr);

			H5Sclose(file_space);

			++g_shape[rank];

			H5Dset_extent(dset, g_shape);
		}

		file_space = H5Dget_space(dset);

		H5_ERROR(H5Sget_simple_extent_dims(file_space, g_shape, nullptr));

		f_begin[rank] = g_shape[rank] - 1;

		m_count[rank] = 1;

		H5_ERROR(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, f_begin, nullptr, m_count, nullptr));

		mem_space = H5Screate_simple(rank + 1, m_shape, nullptr);

		H5_ERROR(H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, m_begin, NULL, m_count, NULL));

	}
	else //   append, extent first dimension; file data has rank dimension
	{

		if (H5Lexists(group_, dsname.c_str(), H5P_DEFAULT) == 0)
		{
			hsize_t max_dims[rank ];

			std::copy(g_shape, g_shape + rank, max_dims);

			max_dims[0] = H5S_UNLIMITED;

			hid_t space = H5Screate_simple(rank , g_shape, max_dims);

			hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);

			H5_ERROR(H5Pset_chunk(dcpl_id, rank , g_shape));

			dset = H5Dcreate(group_, dsname.c_str(), m_type, space, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

			H5_ERROR(H5Sclose(space));

			H5_ERROR(H5Pclose(dcpl_id));

		}
		else
		{

			dset = H5Dopen(group_, dsname.c_str(), H5P_DEFAULT);

			file_space = H5Dget_space(dset);

			int ndims= H5Sget_simple_extent_ndims(file_space);

			assert(ndims==rank);

			H5Sget_simple_extent_dims(file_space, g_shape, nullptr);

			H5Sclose(file_space);

			g_shape[0]+=m_count[0];

			H5Dset_extent(dset, g_shape);
		}

		//		CHECK(name);
		//		CHECK(rank);
		//		CHECK(g_shape[0]) << " " << g_shape[1] << " " << g_shape[2] << " " << g_shape[3];
		//		CHECK(f_begin[0]) << " " << f_begin[1] << " " << f_begin[2] << " " << f_begin[3];
		//		CHECK(m_shape[0]) << " " << m_shape[1] << " " << m_shape[2] << " " << m_shape[3];
		//		CHECK(m_begin[0]) << " " << m_begin[1] << " " << m_begin[2] << " " << m_begin[3];
		//		CHECK(m_count[0]) << " " << m_count[1] << " " << m_count[2] << " " << m_count[3];

		file_space = H5Dget_space(dset);

		H5_ERROR(H5Sget_simple_extent_dims(file_space, g_shape, nullptr));

		f_begin[0] = g_shape[0] - m_count[0];

		H5_ERROR(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, f_begin, nullptr, m_count, nullptr));

		mem_space = H5Screate_simple(rank , m_shape, nullptr);

		H5_ERROR(H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, m_begin, NULL, m_count, NULL));

	}

// create property list for collective dataset write.
	if (GLOBAL_COMM.is_ready())
	{
		hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
		H5_ERROR(H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT));
		H5_ERROR(H5Dwrite(dset,m_type, mem_space, file_space, plist_id, v));
		H5_ERROR(H5Pclose(plist_id));
	}
	else
	{
		H5_ERROR(H5Dwrite(dset,m_type , mem_space, file_space, H5P_DEFAULT, v));
	}

	H5_ERROR(H5Dclose(dset));

	H5_ERROR(H5Sclose(mem_space));

	H5_ERROR(H5Sclose(file_space));

	if (H5Tcommitted(m_type) > 0)
		H5Tclose(m_type);

	return "\"" + GetCurrentPath() + dsname + "\"";
}

void sync_location(hsize_t count[2])
{

	if (!GLOBAL_COMM.is_ready() )
	return;

	auto comm=GLOBAL_COMM.comm();

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

	if (rank == 0)
	buffer.resize(size);

	MPI_Gather(&count[1], 1, m_type.type(), &buffer[0], 1, m_type.type(), 0, comm);

	MPI_Barrier (comm);
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

}

std::string DataStream::pimpl_s::write2(std::string const &name, void const *v, DataType const & data_desc,
        size_t count) const
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

	hsize_t pos[2] = { 0, count };

	sync_location(pos);

	int rank = data_desc.NDIMS + 1;

	hsize_t f_count[rank];
	hsize_t f_begin[rank];
	hsize_t m_count[rank];

	f_begin[0] = pos[0];
	f_count[0] = pos[1];
	m_count[0] = count;

	if (data_desc.NDIMS > 0)
	{
		for (int j = 0; j < data_desc.NDIMS; ++j)
		{

			f_count[1 + j] = data_desc.dimensions_[j];

			f_begin[1 + j] = 0;

			m_count[1 + j] = data_desc.dimensions_[j];

		}
	}

	hid_t m_type = GLOBAL_HDF5_DATA_TYPE_FACTORY.create(data_desc.t_info_);

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
		H5_ERROR(H5Dwrite(dset,m_type, mem_space, file_space, plist_id, v));
		H5_ERROR(H5Pclose(plist_id));
	}
	else
	{
		H5_ERROR(H5Dwrite(dset,m_type , mem_space, file_space, H5P_DEFAULT, v));
	}

	H5_ERROR(H5Dclose(dset));

	H5_ERROR(H5Sclose(mem_space));

	H5_ERROR(H5Sclose(file_space));

	if (H5Tcommitted(m_type) > 0)
		H5Tclose(m_type);

	return "\"" + GetCurrentPath() + dsname + "\"";
}

//=====================================================================================
DataStream::DataStream()
		: pimpl_(new pimpl_s)
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
std::string DataStream::write(std::string const &name, void const *v,

DataType const & datatype,

size_t ndims_or_number,

size_t const *global_begin,

size_t const *global_end,

size_t const *local_outer_begin,

size_t const *local_outer_end,

size_t const *local_inner_begin,

size_t const *local_inner_end,

unsigned int flag

) const
{
	return pimpl_->write(name, v, datatype, ndims_or_number, global_begin,

	global_end,

	local_outer_begin,

	local_outer_end,

	local_inner_begin,

	local_inner_end, flag);
}
}
// namespace simpla
