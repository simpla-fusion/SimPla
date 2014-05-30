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
typename std::enable_if<has_static_member_function_DataTypeDesc<T>::value, MPI_Datatype>::type GetMPIDataType()
{
	MPI_Datatype res;

	return res;
}
template<typename T>
typename std::enable_if<!has_static_member_function_DataTypeDesc<T>::value, MPI_Datatype>::type GetMPIDataType()
{
	return MPI_DATATYPE_NULL;
}

template<typename T> struct MPIPredefineDataType
{
	static MPI_Datatype type() const
	{
		return MPI_DATATYPE_NULL;
	}

};
template<> struct MPIPredefineDataType<int>
{
	static MPI_Datatype type() const
	{
		return MPI_INT;
	}

};
template<> struct MPIPredefineDataType<long>
{
	static MPI_Datatype type() const
	{
		return MPI_LONG;
	}
};

template<> struct MPIPredefineDataType<float>
{
	static MPI_Datatype type() const
	{
		return MPI_FLOAT;
	}
};

template<> struct MPIPredefineDataType<double>
{
	static MPI_Datatype type() const
	{
		return MPI_DOUBLE;
	}
};

template<> struct MPIPredefineDataType<long double>
{
	static MPI_Datatype type() const
	{
		return MPI_LONG_DOUBLE;
	}
};
struct MPIPredefineDataType<std::complex<double>>
{
	static MPI_Datatype type() const
	{
		return MPI_2DOUBLE_COMPLEX;
	}
};

struct MPIPredefineDataType<std::complex<float>>
{
	static MPI_Datatype type() const
	{
		return MPI_2COMPLEX;
	}
};

template<typename TV, int NDIMS>
MPI_Datatype MPICreateArray(MPIDataType<TV> const& old_type, nTuple<NDIMS, size_t> const &outer,
        nTuple<NDIMS, size_t> const &inner, nTuple<NDIMS, size_t> const &start, int array_order_ = MPI_ORDER_C)
{

	MPI_Datatype data_type;

	MPI_Type_create_subarray(NDIMS, &outer[0], &inner[0], &start[0], array_order_, old_type.type(), &data_type);

	return data_type;
}
template<int N, typename TV, int NDIMS>
MPI_Datatype MPICreateArray(MPIDataType<nTuple<N, TV>> const& old_type, nTuple<NDIMS, size_t> const &outer,
        nTuple<NDIMS, size_t> const &inner, nTuple<NDIMS, size_t> const &start, int array_order_ = MPI_ORDER_C)
{
	nTuple<NDIMS + 1, size_t> const &outer1;
	nTuple<NDIMS + 1, size_t> const &inner1;
	nTuple<NDIMS + 1, size_t> const &start1;

	for (int i = 0; i < NDIMS; ++i)
	{
		outer1[i] = outer[i];
		inner1[i] = inner[i];
		start1[i] = start[i];
	}
	outer1[NDIMS] = N;
	inner1[NDIMS] = inner[N];
	start1[NDIMS] = start[N];

	return MPICreateArray(MPIDataType<TV>(), outer1, inner1, start1, array_order_);
}
} // namespace _impl
template<typename T>
struct MPIDataType
{
	MPI_Datatype type_;

	MPIDataType()
			: type_(_impl::MPIPredefineDataType<T>::type())
	{
	}

	template<typename ...Args>
	MPIDataType(MPI_Comm comm, Args const & ... args)
			: type_(MPI_DATATYPE_NULL)
	{
		type_ = _impl::MPICreateArray(MPIDataType<T>(), std::forward<Args const &>(args)...);
		MPI_Type_commit(&type_);

	}
	~MPIDataType()
	{
		MPI_Type_free(&type_);
	}

	MPI_Datatype const & type(...) const
	{
		return type_;
	}

};

//template<typename TL, typename TR>
//struct MPIDataType<std::pair<TL, TR> >
//{
//	typedef std::pair<TL, TR> value_type;
//	MPI_Datatype type_;
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
//	MPI_Datatype type() const
//	{
//		return type_;
//	}
//};

}// namespace simpla

#endif /* MPI_DATATYPE_H_ */
