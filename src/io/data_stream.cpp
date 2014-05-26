/*
 * data_stream.cpp
 *
 *  Created on: 2013年12月12日
 *      Author: salmon
 */

#include "data_stream.h"
#include "../parallel/parallel.h"

namespace simpla
{

void DataStream::OpenGroup(std::string const & gname)
{
	if (gname == "")
		return;

	hid_t h5fg = file_;

	CloseGroup();

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
		ERROR << "Can not open group " << grpname_ << " in file " << prefix_;
	}

}

void DataStream::OpenFile(std::string const &fname)
{

	CloseFile();
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

	hid_t plist_id = H5P_DEFAULT;

	if (GLOBAL_COMM.IsInitilized())
	{
		plist_id = H5Pcreate(H5P_FILE_ACCESS);
		H5Pset_fapl_mpio(plist_id, GLOBAL_COMM.GetComm(), GLOBAL_COMM.GetInfo());
	}

	H5_ERROR( file_ = H5Fcreate(filename_.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id));

	H5Pclose(plist_id);

	if (file_ < 0)
	{
		ERROR << "Create HDF5 file " << filename_ << " failed!" << std::endl;
	}

	OpenGroup("");
}

void DataStream::CloseGroup()
{
	if (group_ > 0)
	{
		H5Gclose(group_);
	}
	group_ = -1;
}
void DataStream::CloseFile()
{
	CloseGroup();
	if (file_ > 0)
	{
		H5Fclose(file_);
	}
	file_ = -1;
}

std::string DataStream::WriteHDF5(void const *v, std::string const &name, hid_t mdtype, int rank,
        hsize_t const *global_dims, hsize_t const *offset, hsize_t const *local_dims, hsize_t const *start,
        hsize_t const *counts, hsize_t const *strides, hsize_t const *blocks) const
{

	if (v == nullptr)
	{
		WARNING << name << " is empty!";
	}

	if (group_ <= 0)
	{
		WARNING << "HDF5 file is not opened! No data is saved!";
	}

	std::string dsname = name;

	hid_t dset;

	hid_t file_space, mem_space;

	if (!is_compact_storable_)
	{

		dsname = name +

		AutoIncrease([&](std::string const & s )->bool
		{
			return H5Lexists(group_, (name + s ).c_str(), H5P_DEFAULT) > 0;
		}, 0, 4);

		file_space = H5Screate_simple(rank, global_dims, nullptr);

		dset = H5Dcreate(group_, dsname.c_str(), mdtype, file_space,
		H5P_DEFAULT,
		H5P_DEFAULT, H5P_DEFAULT);

		H5_ERROR(H5Sclose(file_space));

		H5_ERROR(H5Fflush(group_, H5F_SCOPE_GLOBAL));

		file_space = H5Dget_space(dset);
		H5_ERROR(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, NULL, counts, NULL));

		mem_space = H5Screate_simple(rank, local_dims, NULL);
		H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, start, NULL, counts,
		NULL);

	}
	else
	{
		++rank;

		if (H5Lexists(group_, dsname.c_str(), H5P_DEFAULT) == 0)
		{
			hsize_t chunk_dims[rank];
			hsize_t max_dims[rank];

			std::copy(global_dims, global_dims + rank - 1, chunk_dims + 1);
			std::copy(global_dims, global_dims + rank - 1, max_dims + 1);

			chunk_dims[0] = 1;
			max_dims[0] = H5S_UNLIMITED;

			hid_t space = H5Screate_simple(rank, chunk_dims, max_dims);

			hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);

			H5_ERROR(H5Pset_chunk(dcpl_id, rank, chunk_dims));

			dset = H5Dcreate(group_, dsname.c_str(), mdtype, space, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

			H5_ERROR(H5Sclose(space));

			H5_ERROR(H5Pclose(dcpl_id));

		}
		else
		{

			dset = H5Dopen(group_, dsname.c_str(), H5P_DEFAULT);

			hsize_t fdims[rank];

			file_space = H5Dget_space(dset);

			H5Sget_simple_extent_dims(file_space, fdims, nullptr);

			H5Sclose(file_space);

			++fdims[0];

			H5Dset_extent(dset, fdims);
		}

		file_space = H5Dget_space(dset);

		hsize_t counts_[rank];

		hsize_t offset_[rank];

		H5_ERROR(H5Sget_simple_extent_dims(file_space, counts_, nullptr));

		offset_[0] = counts_[0] - 1;

		counts_[0] = 1;

		std::copy(offset, offset + rank - 1, offset_ + 1);

		std::copy(counts, counts + rank - 1, counts_ + 1);

		H5_ERROR(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset_, nullptr, counts_, nullptr));

		hsize_t local_dims_[rank];

		hsize_t start_[rank];

		std::copy(local_dims, local_dims + rank - 1, local_dims_ + 1);

		std::copy(start, start + rank - 1, start_ + 1);

		local_dims_[0] = 1;

		start_[0] = 0;

		mem_space = H5Screate_simple(rank, local_dims_, nullptr);

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

	return "\"" + GetCurrentPath() + name + "\"";
}

}  // namespace simpla
