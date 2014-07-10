/*
 * mpi_datatype.h
 *
 *  created on: 2014-5-26
 *      Author: salmon
 */

#ifndef MPI_DATATYPE_H_
#define MPI_DATATYPE_H_

#include <mpi.h>
#include <complex>

#include <utility>
#include <vector>
#include "../utilities/ntuple.h"
#include "../utilities/sp_type_traits.h"
#include "../utilities/data_type.h"

namespace simpla
{
namespace _impl
{

bool GetMPIType(DataType const & datatype_desc, MPI_Datatype * new_type);

}  // namespace _impl

/**
 *  \ingroup MPI
 *  \brief MPI convert C++ data type to mpi data type
 *  \todo change to pimpl
 */
template<typename T>
struct MPIDataType
{
	MPI_Datatype type_;
	bool is_commited_ = false;
	static constexpr unsigned int MAX_NTUPLE_RANK = 10;
	MPIDataType()
			: is_commited_(_impl::GetMPIType(DataType::create<T>(), &type_))
	{
	}
	template<unsigned int NDIMS, typename TI>
	MPIDataType(nTuple<NDIMS, TI> const &outer, nTuple<NDIMS, TI> const &inner, nTuple<NDIMS, TI> const &start,
	        unsigned int array_order_ =
	        MPI_ORDER_C)
	{
		const int v_ndims = nTupleTraits<T>::NDIMS;

		int outer1[NDIMS + v_ndims];
		int inner1[NDIMS + v_ndims];
		int start1[NDIMS + v_ndims];
		for (int i = 0; i < NDIMS; ++i)
		{
			outer1[i] = outer[i];
			inner1[i] = inner[i];
			start1[i] = start[i];
		}

		nTupleTraits<T>::get_dimensions(outer1 + NDIMS);
		nTupleTraits<T>::get_dimensions(inner1 + NDIMS);
		for (int i = 0; i < v_ndims; ++i)
		{
			start1[NDIMS + i] = 0;
		}

		MPI_Type_create_subarray(NDIMS + v_ndims, outer1, inner1, start1, array_order_,
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

}  // namespace simpla

#endif /* MPI_DATATYPE_H_ */
