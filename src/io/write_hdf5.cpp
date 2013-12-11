/*
 * write_hdf5.cpp
 *
 *  Created on: 2013年12月3日
 *      Author: salmon
 */
#include "write_hdf5.h"
namespace simpla
{
namespace HDF5
{

void HDF5Write(hid_t grp, std::string const & name, void const * v,
		hid_t mdtype, int n, size_t * dims, bool append_enable)
{
	std::string dname = (name == "" ? "unnamed" : name) +

	AutoIncrease([&](std::string const & s )->bool
	{
		return H5Gget_objinfo(grp, (dname + s ).c_str(),
				false, nullptr) < 0;
	}, 0);

	if (!append_enable)
	{

		hsize_t mdims[n];

		std::copy(dims, dims + n, mdims);

		hid_t dspace = H5Screate_simple(n, mdims, mdims);
		hid_t dset = H5Dcreate(grp, dname.c_str(), mdtype, dspace, H5P_DEFAULT,
		H5P_DEFAULT, H5P_DEFAULT);

		H5Dwrite(dset, mdtype, dspace, dspace, H5P_DEFAULT, v);

		H5Dclose(dset);
		H5Dclose(dspace);

	}
	else
	{
		hid_t dset;
		hid_t fspace;
		hid_t mspace;

		int ndims = n + 1;

		hsize_t start[ndims];
		std::fill(start, start + ndims, 0);

		hsize_t mdims[ndims];
		mdims[0] = 1;
		std::copy(dims, dims + n, mdims + 1);
		hsize_t fdims[ndims];
		fdims[0] = H5S_UNLIMITED;
		std::copy(dims, dims + n, fdims + 1);

		mspace = H5Screate_simple(ndims, mdims, mdims);

		if (H5LTfind_dataset(grp, dname.c_str()))
		{
			dset = H5Dopen1(grp, dname.c_str());
			fspace = H5Dget_space(dset);
			H5Sset_extent_simple(fspace, ndims, fdims, fdims);
		}
		else
		{
			hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
			H5Pset_chunk(plist, ndims, mdims);
			fspace = H5Screate_simple(ndims, mdims, fdims);
			dset = H5Dcreate(grp, dname.c_str(), mdtype, fspace, plist,
			H5P_DEFAULT, H5P_DEFAULT);
			fdims[0] = 0;
			H5Pclose(plist);
		}

		start[0] = fdims[0];

		++fdims[0];

		H5Dextend(dset, fdims);

		H5Fflush(grp, H5F_SCOPE_GLOBAL);

		fspace = H5Dget_space(dset);

		H5Sselect_hyperslab(fspace, H5S_SELECT_SET, mdims, start, H5P_DEFAULT,
		H5P_DEFAULT);

		H5Dwrite(dset, mdtype, mspace, fspace, H5P_DEFAULT, v);

		H5Sclose(mspace);
		H5Sclose(fspace);
		H5Dclose(dset);
	}

}

}  // namespace HDF5
}  // namespace simpla
