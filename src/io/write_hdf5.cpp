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
		hid_t const & mdtype, int n, size_t * dims, bool append_enable)
{

	if (!append_enable)
	{

		hsize_t mdims[n];

		std::copy(dims, dims + n, mdims);

		grp

		.createDataSet(name.c_str(), mdtype, H5::DataSpace(n, mdims))

		.write(v, mdtype);

	}
	else
	{
		hid_t dataset;
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

		mspace = H5::DataSpace(ndims, mdims);

		if (H5LTfind_dataset(grp.getLocId(), name.c_str()))
		{
			dataset = grp.openDataSet(name);
			fspace = dataset.getSpace();
			fspace.getSimpleExtentDims(fdims);
		}
		else
		{

			H5::DSetCreatPropList plist;
			plist.setChunk(ndims, mdims);
			fspace = H5::DataSpace(ndims, mdims, fdims);
			dataset = grp.createDataSet(name.c_str(), mdtype, fspace, plist);
			fdims[0] = 0;
		}

		start[0] = fdims[0];

		++fdims[0];

		dataset.extend(fdims);

		grp.flush(H5F_SCOPE_GLOBAL);

		fspace = dataset.getSpace();

		fspace.selectHyperslab(H5S_SELECT_SET, mdims, start);

		dataset.write(v, mdtype, mspace, fspace);

	}

}

}  // namespace HDF5
}  // namespace simpla
