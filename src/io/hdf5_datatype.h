/*
 * hdf5_datatype.h
 *
 *  Created on: 2014年5月26日
 *      Author: salmon
 */

#ifndef HDF5_DATATYPE_H_
#define HDF5_DATATYPE_H_

#include <H5Ipublic.h>
#include <H5LTpublic.h>
#include <H5Tpublic.h>
#include <complex>
#include <type_traits>
#include <utility>

#include "../utilities/utilities.h"
#include "../fetl/ntuple.h"
namespace simpla
{
namespace _impl
{

HAS_STATIC_MEMBER_FUNCTION(DataTypeDesc);

template<typename T>
typename std::enable_if<has_static_member_function_DataTypeDesc<T>::value, hid_t>::type GetH5Type()
{
	hid_t res;
	H5_ERROR(res = H5LTtext_to_dtype(T::DataTypeDesc().c_str(), H5LT_DDL));
	return res;
}
template<typename T>
typename std::enable_if<!has_static_member_function_DataTypeDesc<T>::value, hid_t>::type GetH5Type()
{
	return H5T_OPAQUE;
}

}  // namespace _impl

template<typename T>
struct HDF5DataType
{
	hid_t type(...) const
	{
		return _impl::GetH5Type<T>();
	}
};

template<> struct HDF5DataType<int>
{
	hid_t type() const
	{
		return H5T_NATIVE_INT;
	}
};

template<> struct HDF5DataType<float>
{
	hid_t type() const
	{
		return H5T_NATIVE_FLOAT;
	}
};

template<> struct HDF5DataType<double>
{
	hid_t type() const
	{
		return H5T_NATIVE_DOUBLE;
	}
};

template<> struct HDF5DataType<long double>
{
	hid_t type() const
	{
		return H5T_NATIVE_LDOUBLE;
	}
};
template<typename T> struct HDF5DataType<std::complex<T>>
{
	hid_t type_;
	HDF5DataType()
	{
		type_ = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<T>));
		H5Tinsert(type_, "r", 0, HDF5DataType<T>().type());
		H5Tinsert(type_, "i", sizeof(T), HDF5DataType<T>().type());
	}

	~ HDF5DataType()
	{
		H5Tclose(type_);
	}

	hid_t type() const
	{
		return type_;
	}
};

template<typename TL, typename TR>
struct HDF5DataType<std::pair<TL, TR> >
{
	typedef std::pair<TL, TR> value_type;
	hid_t type_;
	HDF5DataType()
	{
		type_ = H5Tcreate(H5T_COMPOUND, sizeof(value_type));
		H5Tinsert(type_, "first", offsetof(value_type, first), HDF5DataType<TL>().type());
		H5Tinsert(type_, "second", offsetof(value_type, second), HDF5DataType<TR>().type());
	}

	~ HDF5DataType()
	{
		H5Tclose(type_);
	}

	hid_t type() const
	{
		return type_;
	}
};
}  // namespace simpla

#endif /* HDF5_DATATYPE_H_ */
