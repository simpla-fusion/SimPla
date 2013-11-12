/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * IO/HDF5Write.h
 *
 *  Created on: 2011-1-27
 *      Author: salmon
 */

#ifndef SRC_IO_WRITE_HDF5_H_
#define SRC_IO_WRITE_HDF5_H_
#include <H5Cpp.h>
#include "include/simpla_defs.h"
#include "engine/object.h"

namespace simpla
{
namespace io
{

template<typename > struct HDF5DataType;

template<> struct HDF5DataType<int>
{
	H5::DataType operator()()
	{
		return H5::PredType::NATIVE_INT;
	}
};

template<> struct HDF5DataType<float>
{
	H5::DataType operator()()
	{
		return H5::PredType::NATIVE_FLOAT;
	}
};

template<> struct HDF5DataType<double>
{
	H5::DataType operator()()
	{
		return H5::PredType::NATIVE_DOUBLE;
	}
};

template<> struct HDF5DataType<long double>
{
	H5::DataType operator()()
	{
		return H5::PredType::NATIVE_LDOUBLE;
	}
};

template<typename T>
void HDF5Write(H5::Group grp, std::string const & name, T const * v,
		size_t size)
{
	try
	{

		H5::DataType mdtype = HDF5DataType<T>();

		hsize_t extents = size;

		H5::DataSet dataset = grp.createDataSet(name.c_str(), mdtype,
				H5::DataSpace(1, &extents));

		dataset.write(v, mdtype);

	} catch (H5::Exception &e)
	{
		ERROR << "Can not write dataset  to [" << name.c_str()
				<< "]! \nError:  " << e.getDetailMsg();
		throw(e);
	}
}

template<int N, typename T>
void HDF5Write(H5::Group grp, std::string const & name, nTuple<N, T> const * v,
		size_t size)
{
	try
	{

		H5::DataType mdtype = HDF5DataType<T>();

		hsize_t extents[2] =
		{ size, 2 };

		H5::DataSet dataset = grp.createDataSet(name.c_str(), mdtype,
				H5::DataSpace(2, extents));

		dataset.write(v, mdtype);

	} catch (H5::Exception &e)
	{
		ERROR << "Can not write dataset  to [" << name.c_str()
				<< "]! \nError:  " << e.getDetailMsg();
		throw(e);
	}
}

template<typename T>
void HDF5Write(H5::Group grp, std::string const & name,
		std::vector<T> const & v)
{
	HDF5Write(grp, name, v.data(), v.size());
}

template<typename T>
void HDF5Write(H5::Group grp, std::string const & name, T const & v)
{
	// TODO: need optimize

	std::vector<decltype(*v.begin())> tmp(v.begin(), v.end());
	HDF5Write(grp, name, tmp.data, tmp.size());
}

template<typename T>
void HDF5Append(H5::Group grp, std::string const & name, T const *v,
		size_t size)
{
	try
	{

		H5::DataSet dataset;
		H5::DataSpace fspace;
		H5::DataSpace mspace;
		H5::DataType mdtype = HDF5DataType<T>();

		int ndims = 1;
		hsize_t extents = size;

		hsize_t start[2] =
		{ 0, 0 };

		hsize_t mdims[2] =
		{ 1, extents };

		hsize_t fdims[2] =
		{ H5S_UNLIMITED, extents };

		mdims[0] = 1;

		mspace = H5::DataSpace(2, mdims);

		if (H5LTfind_dataset(grp.getLocId(), name.c_str()))
		{
			dataset = grp.openDataSet(name);
			fspace = dataset.getSpace();
			fspace.getSimpleExtentDims(fdims);
		}
		else
		{

			H5::DSetCreatPropList plist;
			plist.setChunk(2, mdims);
			fspace = H5::DataSpace(2, mdims, fdims);
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

	} catch (H5::Exception &e)
	{
		ERROR << "Can not write dataset  to [" << name.c_str()
				<< "]! \nError:  " << e.getDetailMsg();
		throw(e);
	}
}

template<int N, typename T>
void HDF5Append(H5::Group grp, std::string const & name, nTuple<N, T> const *v,
		size_t size)
{
	try
	{

		H5::DataSet dataset;
		H5::DataSpace fspace;
		H5::DataSpace mspace;
		H5::DataType mdtype = HDF5DataType<T>();

		hsize_t extents = size;

		hsize_t start[3] =
		{ 0, 0, 0 };

		hsize_t mdims[3] =
		{ 1, extents, N };

		hsize_t fdims[3] =
		{ H5S_UNLIMITED, extents, N };

		mspace = H5::DataSpace(3, mdims);

		if (H5LTfind_dataset(grp.getLocId(), name.c_str()))
		{
			dataset = grp.openDataSet(name);
			fspace = dataset.getSpace();
			fspace.getSimpleExtentDims(fdims);
		}
		else
		{
			H5::DSetCreatPropList plist;
			plist.setChunk(3, mdims);
			fspace = H5::DataSpace(3, mdims, fdims);
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

	} catch (H5::Exception &e)
	{
		ERROR << "Can not write dataset  to [" << name.c_str()
				<< "]! \nError:  " << e.getDetailMsg();
		throw(e);
	}
}

template<typename T>
void HDF5Append(H5::Group grp, std::string const & name,
		std::vector<T> const & v)
{
	HDF5Append(grp, name, v.data(), v.size());
}
template<typename T>
void HDF5Append(H5::Group grp, std::string const & name, T const & v)
{
	// TODO: need optimize

	HDF5Append(grp, name,
			std::vector<decltype(*v.begin())>(v.begin(), v.end()));
}
} // namespace IO
} // namespace simpla
#endif  // SRC_IO_WRITE_HDF5_H_
