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
#include <vector>

#include "../utilities/type_utilites.h"

namespace simpla
{
namespace _impl
{
template<typename T>
bool GetMPIType(MPI_Datatype * new_type)
{
	bool is_commited = false;
	if (typeid(T) == typeid(int))
	{
		*new_type = MPI_INT;
	}
	else if (typeid(T) == typeid(long))
	{
		*new_type = MPI_LONG;
	}
	else if (typeid(T) == typeid(float))
	{
		*new_type = MPI_FLOAT;
	}
	else if (typeid(T) == typeid(double))
	{
		*new_type = MPI_DOUBLE;
	}
	else if (typeid(T) == typeid(long double))
	{
		*new_type = MPI_LONG_DOUBLE;
	}
	else if (typeid(T) == typeid(std::complex<double>))
	{
		*new_type = MPI_2DOUBLE_COMPLEX;
	}
	else if (typeid(T) == typeid(std::complex<float>))
	{
		*new_type = MPI_2COMPLEX;
	}
	else if (is_nTuple<T>::value)
	{
		// FIXME incomplete implement , need nTuple<N,nTuple<M,TV>>
		typedef typename nTupleTraits<T>::element_type value_type;
		std::vector<int> dims;
		nTupleTraits<T>::GetDims(&dims);

		MPI_Datatype old_type;

		GetMPIType<value_type>(&old_type);

		MPI_Type_contiguous(nTupleTraits<T>::NDIMS, old_type, new_type);

		MPI_Type_commit(new_type);

		is_commited = true;
	}
	else
	{
		MPI_Type_contiguous(sizeof(T) / sizeof(char), MPI_CHAR, new_type);
		MPI_Type_commit(new_type);
		is_commited = true;

	}

	return is_commited;
}

}  // namespace _impl
template<typename T>
struct MPIDataType
{
	MPI_Datatype type_;
	bool is_commited_ = false;
	static constexpr int MAX_NTUPLE_RANK = 10;
	MPIDataType() :
			is_commited_(_impl::GetMPIType<T>(&type_))
	{
	}
	template<int NDIMS>
	MPIDataType(nTuple<NDIMS, size_t> const &outer, nTuple<NDIMS, size_t> const &inner,
			nTuple<NDIMS, size_t> const &start, int array_order_ =
			MPI_ORDER_C)
	{

		std::vector<int> outer1;
		std::vector<int> inner1;
		std::vector<int> start1;
		std::copy(&outer[0], &outer[NDIMS], std::back_inserter(outer1));
		std::copy(&inner[0], &inner[NDIMS], std::back_inserter(inner1));
		std::copy(&start[0], &start[NDIMS], std::back_inserter(start1));

		nTupleTraits<T>::GetDims(&outer1);
		nTupleTraits<T>::GetDims(&inner1);
		for (int i = start1.size(), ie = outer1.size(); i < ie; ++i)
		{
			start1.push_back(0);
		}

		MPI_Type_create_subarray(outer1.size(), &outer1[0], &inner1[0], &start1[0], array_order_,
				MPIDataType<typename nTupleTraits<T>::element_type>().type(), &type_);
		MPI_Type_commit(&type_);
		is_commited_ = true;
	}

	~MPIDataType()
	{
		if (is_commited_)
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
