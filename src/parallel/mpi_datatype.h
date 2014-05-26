/*
 * mpi_datatype.h
 *
 *  Created on: 2014年5月26日
 *      Author: salmon
 */

#ifndef MPI_DATATYPE_H_
#define MPI_DATATYPE_H_

#include <mpi.h>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "../utilities/type_utilites.h"

namespace simpla
{
namespace _impl
{

HAS_STATIC_MEMBER_FUNCTION(MPIDataTypeDesc);

template<typename T>
typename std::enable_if<has_static_member_function_DataTypeDesc<T>::value, hid_t>::type GetMPIDataType()
{
	hid_t res;

	return res;
}
template<typename T>
typename std::enable_if<!has_static_member_function_DataTypeDesc<T>::value, hid_t>::type GetMPIDataType()
{
	return MPI_DATATYPE_NULL;
}

}  // namespace _impl
template<typename T>
struct MPIDataType
{
	hid_t type(...) const
	{
		return _impl::GetMPIDataType<T>();
	}
};

template<> struct MPIDataType<int>
{
	hid_t type() const
	{
		return MPI_INT;
	}
};
template<> struct MPIDataType<long>
{
	hid_t type() const
	{
		return MPI_LONG;
	}
};

template<> struct MPIDataType<float>
{
	hid_t type() const
	{
		return MPI_FLOAT;
	}
};

template<> struct MPIDataType<double>
{
	hid_t type() const
	{
		return MPI_DOUBLE;
	}
};

template<> struct MPIDataType<long double>
{
	hid_t type() const
	{
		return MPI_LONG_DOUBLE;
	}
};
struct MPIDataType<std::complex<double>>
{
	hid_t type() const
	{
		return MPI_2DOUBLE_COMPLEX;
	}
};

struct MPIDataType<std::complex<float>>
{
	hid_t type() const
	{
		return MPI_2COMPLEX;
	}
};

//template<typename TL, typename TR>
//struct MPIDataType<std::pair<TL, TR> >
//{
//	typedef std::pair<TL, TR> value_type;
//	hid_t type_;
//	MPIDataType()
//	{
//
//	}
//
//	~ MPIDataType()
//	{
//
//	}
//
//	hid_t type() const
//	{
//		return type_;
//	}
//};

}// namespace simpla

#endif /* MPI_DATATYPE_H_ */
