/*
 * data_stream.cpp
 *
 *  Created on: 2013年12月12日
 *      Author: salmon
 */

#include "data_stream.h"

extern "C"
{
#include "hdf5.h"
#include "hdf5_hl.h"

}
#include "hdf5_datatype.h"
#include "../parallel/parallel.h"

namespace simpla
{

struct DataStream::pimpl_s
{
	hid_t file_;
	hid_t group_;
};
DataStream::DataStream() :
		prefix_("simpla_unnamed"), filename_("unnamed"), grpname_(""),

		suffix_width_(4),

		LIGHT_DATA_LIMIT_(20),

		enable_compact_storable_(false),

		enable_xdmf_(false),

		pimpl_(new pimpl_s(
		{ -1, -1 }))

{
	hid_t error_stack = H5Eget_current_stack();
	H5Eset_auto(error_stack, NULL, NULL);
}

DataStream::~DataStream()
{
	Close();
	delete pimpl_;
}

void DataStream::OpenGroup(std::string const & gname)
{
	if (gname == "")
		return;

	hid_t h5fg = pimpl_->file_;

	CloseGroup();

	if (gname[0] == '/')
	{
		grpname_ = gname;
	}
	else
	{
		grpname_ += gname;
		if (pimpl_->group_ > 0)
			h5fg = pimpl_->group_;
	}

	if (grpname_[grpname_.size() - 1] != '/')
	{
		grpname_ = grpname_ + "/";
	}

	auto res = H5Lexists(h5fg, grpname_.c_str(), H5P_DEFAULT);

	if (grpname_ == "/" || res != 0)
	{
		H5_ERROR(pimpl_->group_ = H5Gopen(h5fg, grpname_.c_str(), H5P_DEFAULT));
	}
	else
	{
		H5_ERROR(pimpl_->group_ = H5Gcreate(h5fg, grpname_.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
	}
	if (pimpl_->group_ <= 0)
	{
		ERROR << "Can not open group " << grpname_ << " in file " << prefix_;
	}

}

void DataStream::OpenFile(std::string const &fname)
{

	CloseFile();

	MPI_Barrier(GLOBAL_COMM.GetComm());

	if (fname != "")
		prefix_ = fname;

	if (fname.size() > 3 && fname.substr(fname.size() - 3) == ".h5")
	{
		prefix_ = fname.substr(0, fname.size() - 3);
	}

	/// @TODO auto mkdir directory

	filename_ = prefix_ +

	AutoIncrease(

	[&](std::string const & suffix)->bool
	{
		std::string fname=(prefix_+suffix);
		return
		fname==""
		|| *(fname.rbegin())=='/'
		|| (CheckFileExists(fname + ".h5"));
	}

	) + ".h5";

	MPI_Barrier(GLOBAL_COMM.GetComm());

	hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

	H5Pset_fapl_mpio(plist_id, GLOBAL_COMM.GetComm(), GLOBAL_COMM.GetInfo());

	H5_ERROR(pimpl_->file_ = H5Fcreate(filename_.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id));

	H5Pclose(plist_id);

	if (pimpl_->file_ < 0)
	{
		ERROR << "Create HDF5 file " << filename_ << " failed!" << std::endl;
	}

	OpenGroup("");
}

void DataStream::CloseGroup()
{
	if (pimpl_->group_ > 0)
	{
		H5Gclose(pimpl_->group_);
	}
	pimpl_->group_ = -1;
}
void DataStream::CloseFile()
{
	CloseGroup();

	if (pimpl_->file_ > 0)
	{
		H5Fclose(pimpl_->file_);
	}
	pimpl_->file_ = -1;
}

std::string DataStream::WriteHDF5(std::string const &name, void const *v, DataTypeDesc const & mdtype_s,

int rank,

size_t const *p_global_dims,

size_t const *p_local_outer_start,

size_t const *p_local_outer_count,

size_t const *p_local_inner_start,

size_t const *p_local_inner_count) const
{
	if (v == nullptr)
	{
		WARNING << name << " is empty!";
	}

	if (pimpl_->group_ <= 0)
	{
		WARNING << "HDF5 file is not opened! No data is saved!";
	}

	hsize_t global_dims[rank + 2];
	hsize_t local_outer_start[rank + 2];
	hsize_t local_outer_count[rank + 2];
	hsize_t local_inner_start[rank + 2];
	hsize_t local_inner_count[rank + 2];

	std::copy(p_global_dims, p_global_dims + rank, global_dims);
	std::copy(p_local_outer_start, p_local_outer_start + rank, local_outer_start);
	std::copy(p_local_outer_count, p_local_outer_count + rank, local_outer_count);
	std::copy(p_local_inner_start, p_local_inner_start + rank, local_inner_start);
	std::copy(p_local_inner_count, p_local_inner_count + rank, local_inner_count);

	std::string mdtype_str = mdtype_s.type_name;
	if (mdtype_s.array_length > 1)
	{
		global_dims[rank] = mdtype_s.array_length;
		local_outer_start[rank] = 0;
		local_outer_count[rank] = mdtype_s.array_length;
		local_inner_start[rank] = 0;
		local_inner_count[rank] = mdtype_s.array_length;

		mdtype_str = mdtype_s.sub_type_name;
		++rank;
	}

//
//	CHECK(name);
//	CHECK(rank);
//	CHECK(global_dims[0]) << " " << global_dims[1] << " " << global_dims[2] << " " << global_dims[3];
//	CHECK(local_outer_start[0]) << " " << local_outer_start[1] << " " << local_outer_start[2] << " "
//			<< local_outer_start[3];
//	CHECK(local_outer_count[0]) << " " << local_outer_count[1] << " " << local_outer_count[2] << " "
//			<< local_outer_count[3];
//	CHECK(local_inner_start[0]) << " " << local_inner_start[1] << " " << local_inner_start[2] << " "
//			<< local_inner_start[3];
//	CHECK(local_inner_count[0]) << " " << local_inner_count[1] << " " << local_inner_count[2] << " "
//			<< local_inner_count[3];

	hid_t mdtype;

	H5_ERROR(mdtype = H5LTtext_to_dtype(mdtype_str.c_str(), H5LT_DDL));

	std::string dsname = name;

	hid_t dset;

	hid_t file_space, mem_space;

	if (!enable_compact_storable_)
	{

		dsname = name +

		AutoIncrease([&](std::string const & s )->bool
		{
			return H5Lexists(pimpl_->group_, (name + s ).c_str(), H5P_DEFAULT) > 0;
		}, 0, 4);

		file_space = H5Screate_simple(rank, global_dims, nullptr);

		dset = H5Dcreate(pimpl_->group_, dsname.c_str(), mdtype, file_space,
		H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

		H5_ERROR(H5Sclose(file_space));

		H5_ERROR(H5Fflush(pimpl_->group_, H5F_SCOPE_GLOBAL));

		file_space = H5Dget_space(dset);

		H5_ERROR(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, local_inner_start, NULL, local_inner_count, NULL));

		mem_space = H5Screate_simple(rank, local_outer_count, NULL);

		for (int i = 0; i < rank; ++i)
		{
			local_inner_start[i] -= local_outer_start[i];
		}

		H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, local_inner_start, NULL, local_inner_count,
		NULL);

	}
	else
	{

		if (H5Lexists(pimpl_->group_, dsname.c_str(), H5P_DEFAULT) == 0)
		{
			hsize_t chunk_dims[rank + 1];
			hsize_t max_dims[rank + 1];

			std::copy(global_dims, global_dims + rank, chunk_dims + 1);
			std::copy(global_dims, global_dims + rank, max_dims + 1);

			chunk_dims[0] = 1;
			max_dims[0] = H5S_UNLIMITED;

			hid_t space = H5Screate_simple(rank + 1, chunk_dims, max_dims);

			hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);

			H5_ERROR(H5Pset_chunk(dcpl_id, rank + 1, chunk_dims));

			dset = H5Dcreate(pimpl_->group_, dsname.c_str(), mdtype, space, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

			H5_ERROR(H5Sclose(space));

			H5_ERROR(H5Pclose(dcpl_id));

		}
		else
		{

			dset = H5Dopen(pimpl_->group_, dsname.c_str(), H5P_DEFAULT);

			hsize_t fdims[rank + 1];

			file_space = H5Dget_space(dset);

			H5Sget_simple_extent_dims(file_space, fdims, nullptr);

			H5Sclose(file_space);

			++fdims[0];

			H5Dset_extent(dset, fdims);
		}

		file_space = H5Dget_space(dset);

		hsize_t counts_[rank + 1];

		hsize_t offset_[rank + 1];

		H5_ERROR(H5Sget_simple_extent_dims(file_space, counts_, nullptr));

		offset_[0] = counts_[0] - 1;

		counts_[0] = 1;

		std::copy(local_outer_start, local_outer_start + rank, offset_ + 1);

		std::copy(local_inner_count, local_inner_count + rank, counts_ + 1);

		H5_ERROR(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset_, nullptr, counts_, nullptr));

		hsize_t local_dims_[rank + 1];

		hsize_t start_[rank + 1];

		std::copy(local_outer_count, local_outer_count + rank, local_dims_ + 1);

		std::copy(local_inner_start, local_inner_start + rank, start_ + 1);

		local_dims_[0] = 1;

		start_[0] = 0;

		mem_space = H5Screate_simple(rank + 1, local_dims_, nullptr);

		H5_ERROR(H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, start_, NULL, counts_, NULL));

	}

	// Create property list for collective dataset write.
	if (GLOBAL_COMM.IsInitilized())
	{
		hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
		H5_ERROR(H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE));
		H5_ERROR(H5Dwrite(dset, mdtype, mem_space, file_space, plist_id, v));
		H5_ERROR(H5Pclose(plist_id));
	}
	else
	{
		H5_ERROR(H5Dwrite(dset, mdtype, mem_space, file_space, H5P_DEFAULT, v));
	}

	H5_ERROR(H5Dclose(dset));

	H5_ERROR(H5Sclose(mem_space));

	H5_ERROR(H5Sclose(file_space));

	H5_ERROR(H5Tclose(mdtype));

	return "\"" + GetCurrentPath() + name + "\"";
}

}  // namespace simpla
