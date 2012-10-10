/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * io/write_hdf5.cpp
 *
 *  Created on: 2011-1-27
 *      Author: salmon
 */
#include "write_hdf5.h"
#include <H5Cpp.h>
#include <hdf5_hl.h>
#include <algorithm>
#include <list>
#include <string>
#include <vector>

#include "include/simpla_defs.h"
#include "engine/object.h"
namespace simpla
{
namespace io
{

//void HDF5Append(H5::Group grp, std::string const & name,
//		const Object::Holder obj)
//{
//	try
//	{
//
//		H5::DataSet dataset;
//		H5::DataSpace fspace;
//		H5::DataSpace mspace;
//		H5::DataType mdtype(
//				H5LTtext_to_dtype(obj->get_element_type_desc().c_str(),
//						H5LT_DDL));
//
//		int ndims = obj->get_dimensions();
//
//		size_t dims[ndims + 1];
//
//		hsize_t start[ndims + 1];
//		hsize_t mdims[ndims + 1];
//		hsize_t fdims[ndims + 1];
//
//		obj->get_dimensions(dims);
//
//		mdims[0] = 1;
//		mspace = H5::DataSpace(ndims + 1, &mdims[0]);
//
//		if (H5LTfind_dataset(grp.getLocId(), name.c_str()))
//		{
//			dataset = grp.openDataSet(name);
//			fspace = dataset.getSpace();
//			fspace.getSimpleExtentDims(&fdims[0]);
//		}
//		else
//		{
//			std::copy(mdims, mdims + ndims + 1, fdims);
//			fdims[0] = H5S_UNLIMITED;
//			H5::DSetCreatPropList plist;
//			plist.setChunk(ndims + 1, &mdims[0]);
//			fspace = H5::DataSpace(ndims + 1, &mdims[0], &fdims[0]);
//			dataset = grp.createDataSet(name.c_str(), mdtype, fspace, plist);
//			fdims[0] = 0;
//		}
//
//		std::fill(start + 1, start + ndims + 1, 0);
//		start[0] = fdims[0];
//		++fdims[0];
//		dataset.extend(&fdims[0]);
//		grp.flush(H5F_SCOPE_GLOBAL);
//		fspace = dataset.getSpace();
//		fspace.selectHyperslab(H5S_SELECT_SET, &mdims[0], &start[0]);
//
//		dataset.write(obj->get_data(), mdtype, mspace, fspace);
//	} catch (H5::Exception &e)
//	{
//		ERROR << "Can not write dataset  to [" << name.c_str()
//				<< "]! \nError:  " << e.getDetailMsg();
//		throw(e);
//	}
//
//}

void WriteHDF5(const Object::Holder obj, H5::Group grp,
		std::string const & name)
{
	try
	{

		H5::DataType mdtype(
				H5LTtext_to_dtype(obj->get_element_type_desc().c_str(),
						H5LT_DDL));

		int ndims = obj->get_dimensions();

		size_t dims[ndims];

		hsize_t mdims[ndims];

		obj->get_dimensions(dims);

		std::copy(dims, dims + ndims, mdims);

		H5::DataSet dataset = grp.createDataSet(name.c_str(), mdtype,
				H5::DataSpace(ndims, mdims));

		dataset.write(obj->get_data(), mdtype);

//		HDF5AddAttribute(dataset, obj);

	} catch (H5::Exception &e)
	{
		ERROR << "Can not write dataset  to [" << name.c_str()
				<< "]! \nError:  " << e.getDetailMsg();
		throw(e);
	}

}

void HDF5AddAttribute(H5::DataSet dataset, const Object::Holder obj)
{

}
} // namespace IO
} // namespace simpla
