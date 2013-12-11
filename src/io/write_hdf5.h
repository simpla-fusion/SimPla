/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * IO/HDF5Write.h
 *
 *  Created on: 2011-1-27
 *      Author: salmon
 */

#ifndef SRC_IO_WRITE_HDF5_H_
#define SRC_IO_WRITE_HDF5_H_
#include "../utilities/log.h"
#include "../simpla_defs.h"
#include "../fetl/ntuple.h"
#include <H5Cpp.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace simpla
{
namespace HDF5
{

template<typename > struct HDF5DataType;

template<> struct HDF5DataType<int>
{
	hid_t type()
	{
		return H5T_NATIVE_INT;
	}
};

template<> struct HDF5DataType<float>
{
	hid_t type()
	{
		return H5T_NATIVE_FLOAT;
	}
};

template<> struct HDF5DataType<double>
{
	hid_t type()
	{
		return H5T_NATIVE_DOUBLE;
	}
};

template<> struct HDF5DataType<long double>
{
	hid_t type()
	{
		return H5T_NATIVE_LDOUBLE;
	}
};

void HDF5Write(hid_t &grp, std::string const & name, void const * v,
		hid_t const &mdtype, int n, size_t * dims, bool append_enable);

template<typename TV>
inline void HDF5Write(hid_t & grp, std::string const & name,
		std::vector<TV> const &v, int n = 0, size_t * d = nullptr,
		bool append_enable = false)
{
	size_t dims[n + 2];

	hid_t mdtype;

	if (n == 0 || d == nullptr)
	{
		dims[0] = v.size();
		n = 1;
	}
	else
	{
		std::copy(d, d + n, dims);
	}
	if (is_nTuple<TV>::value)
	{
		dims[n] = nTupleTraits<TV>::NUM_OF_DIMS;
		++n;
	}

	HDF5Write(grp, name, reinterpret_cast<void const*>(&v[0]),
			HDF5DataType<typename nTupleTraits<TV>::value_type>().type(), n,
			dims, append_enable);

}

//
//template<int N, typename T>
//void HDF5Write(H5::Group &grp, std::string const & name, nTuple<N, T> const * v,
//		int n, size_t * d, bool append_enable)
//{
//
//	try
//	{
//		H5::DataType mdtype = HDF5DataType<T>().type();
//
//		if (!append_enable)
//		{
//
//			hsize_t mdims[n];
//
//			d[n] = N;
//
//			std::copy(d, d + n, mdims);
//
//			grp
//
//			.createDataSet(name.c_str(), mdtype, H5::DataSpace(n + 1, mdims))
//
//			.write(v, mdtype);
//
//		}
////		H5::DataType mdtype = HDF5DataType<T>().type();
////		if (!append_enable)
////		{
////
////			hsize_t dims[n + 1];
////			std::copy(d, d + n, dims);
////			d[n] = N;
////
////			H5::DataSpace mdspace(n + 1, dims);
////			H5::DataSet ds = grp.createDataSet(name.c_str(), mdtype, mdspace);
////			CHECK(n);
////			CHECK(d[0]);
////			CHECK(d[1]);
////			CHECK(v[0]);
////			ds.write(v, mdtype, mdspace);
////			CHECK("SDFdsfS");
////		}
//
//		else
//		{
//
//			H5::DataSet dataset;
//			H5::DataSpace fspace;
//			H5::DataSpace mspace;
//
//			int ndims = n + 2;
//
//			hsize_t start[ndims];
//			std::fill(start, start + ndims, 0);
//
//			hsize_t mdims[ndims];
//			mdims[0] = 1;
//			mdims[ndims - 1] = N;
//			std::copy(d, d + n, mdims + 1);
//			hsize_t fdims[ndims];
//			fdims[0] = H5S_UNLIMITED;
//			fdims[ndims - 1] = N;
//			std::copy(d, d + n, fdims + 1);
//
//			mspace = H5::DataSpace(ndims, mdims);
//
//			if (H5LTfind_dataset(grp.getLocId(), name.c_str()))
//			{
//				dataset = grp.openDataSet(name);
//				fspace = dataset.getSpace();
//				fspace.getSimpleExtentDims(fdims);
//			}
//			else
//			{
//				H5::DSetCreatPropList plist;
//				plist.setChunk(ndims, mdims);
//				fspace = H5::DataSpace(ndims, mdims, fdims);
//				dataset = grp.createDataSet(name.c_str(), mdtype, fspace,
//						plist);
//				fdims[0] = 0;
//			}
//
//			start[0] = fdims[0];
//
//			++fdims[0];
//
//			dataset.extend(fdims);
//
//			grp.flush(H5F_SCOPE_GLOBAL);
//
//			fspace = dataset.getSpace();
//
//			fspace.selectHyperslab(H5S_SELECT_SET, mdims, start);
//
//			dataset.write(v, mdtype, mspace, fspace);
//
//		}
//	} catch (H5::Exception &e)
//	{
//		ERROR << "Can not write dataset  to [" << name.c_str()
//				<< "]! \nError:  " << e.getDetailMsg();
//		throw(e);
//	}
//}

//template<typename TV, typename ...Args>
//inline void HDF5Write(H5::Group grp, std::string const & name, TV const &v,
//		Args ...args)
//{
//	typedef typename std::remove_reference<
//			typename std::remove_const<decltype(*v.begin())>::type>::type value_type;
//
//	HDF5Append(grp, name, std::vector<value_type>(v.begin(), v.end()),
//			std::forward<Args>(args)...);
//}

}// namespace HDF5
} // namespace simpla
#endif  // SRC_IO_WRITE_HDF5_H_
