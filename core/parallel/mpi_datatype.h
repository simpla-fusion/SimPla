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
#include "../data_structure/data_type.h"

namespace simpla
{
namespace _impl
{

bool GetMPIType(DataType const & datatype_desc, MPI_Datatype * new_type);

}  // namespace _impl

/**
 *  \ingroup MPI
 *  \brief MPI convert C++ data type to mpi data type
 */
struct MPIDataType
{
	MPI_Datatype type_ = MPI_DATATYPE_NULL;
	bool is_commited_ = false;
	static constexpr unsigned int MAXnTuple_RANK = 10;

	MPIDataType()
	{
	}

	~MPIDataType()
	{
		if (is_commited_)
			MPI_Type_free(&type_);
	}

	static MPIDataType create(DataType const &);

	static MPIDataType create(DataType const & data_type, unsigned int NDIMS, size_t const *outer, size_t const * inner,
	        size_t const * start, bool c_order_array = true);

	template<typename T, typename ...Others>
	static MPIDataType create(Others && ... others)
	{
		return create(DataType::create<T>(), std::forward<Others>(others)...);
	}

	MPI_Datatype const & type(...) const
	{
		return type_;
	}

};

}  // namespace simpla

#endif /* MPI_DATATYPE_H_ */
