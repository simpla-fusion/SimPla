/*
 * data_stream.cpp
 *
 *  Created on: 2013年12月12日
 *      Author: salmon
 */

#include "data_stream.h"
namespace simpla
{

std::string HDF5Write(hid_t grp, void const *v, std::string const &name,
        hid_t mdtype, int rank, size_t const *dims, bool APPEND)
{
	if (grp <= 0)
	{
		WARNING << "HDF5 file is not opened! No data is saved!";
		return "";
	}
	std::string dsname = name;

	if (!APPEND)
	{

		dsname = name +

		AutoIncrease([&](std::string const & s )->bool
		{
			return H5Lexists(grp, (name + s ).c_str(), H5P_DEFAULT) > 0;
		}, 0, 4);

		hsize_t mdims[rank];

		std::copy(dims, dims + rank, mdims);

		hid_t dspace = H5Screate_simple(rank, mdims, nullptr);

		hid_t dset = H5Dcreate(grp, dsname.c_str(), mdtype, dspace, H5P_DEFAULT,
		H5P_DEFAULT, H5P_DEFAULT);

		H5Dwrite(dset, mdtype, dspace, dspace, H5P_DEFAULT, v);

		H5_ERROR(H5Dclose(dset));
		H5_ERROR(H5Sclose(dspace));

	}
	else
	{
		int ndims = rank + 1;

		hsize_t chunk_dims[ndims];
		chunk_dims[0] = 1;
		std::copy(dims, dims + rank, chunk_dims + 1);

		if (H5Lexists(grp, dsname.c_str(), H5P_DEFAULT) == 0)
		{
			hid_t dset;

			hsize_t max_dims[ndims];

			std::copy(chunk_dims, chunk_dims + ndims, max_dims);

			max_dims[0] = H5S_UNLIMITED;

			hid_t fspace = H5Screate_simple(ndims, chunk_dims, max_dims);

			hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);

			H5Pset_chunk(dcpl_id, ndims, chunk_dims);

			dset = H5Dcreate(grp, dsname.c_str(), mdtype, fspace, H5P_DEFAULT,
			        dcpl_id, H5P_DEFAULT);

			H5Dwrite(dset, mdtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, v);

			H5Sclose(fspace);

			H5Dclose(dset);

			H5Pclose(dcpl_id);

			H5Fflush(grp, H5F_SCOPE_GLOBAL);

		}
		else
		{

			hid_t dset = H5Dopen(grp, dsname.c_str(), H5P_DEFAULT);

			hid_t fspace = H5Dget_space(dset);

			hsize_t fdims[ndims];

			H5Sget_simple_extent_dims(fspace, fdims, nullptr);

			H5Sclose(fspace);

			hsize_t offset[ndims];

			std::fill(offset, offset + ndims, 0);

			offset[0] = fdims[0];

			++fdims[0];

			H5Dset_extent(dset, fdims);

			fspace = H5Dget_space(dset);

			H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, nullptr,
			        chunk_dims, nullptr);

			hid_t mspace = H5Screate_simple(ndims, chunk_dims, nullptr);

			H5Dwrite(dset, mdtype, mspace, fspace, H5P_DEFAULT, v);

			H5Dclose(dset);

			H5Sclose(mspace);

			H5Sclose(fspace);

			H5Fflush(grp, H5F_SCOPE_GLOBAL);

		}

	}

	return dsname;

}

}  // namespace simpla
