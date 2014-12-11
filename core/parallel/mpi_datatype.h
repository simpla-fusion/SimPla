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
 *  \brief MPI convert C++ data type and data space to mpi data type
 */
struct MPIDataType
{

	MPIDataType();

	~MPIDataType();

	static MPIDataType create(DataType const &);

	static MPIDataType create(DataType const & data_type, unsigned int ndims,
			size_t const * dims, size_t const * offset, size_t const * stride,
			size_t const * count, size_t const * block, bool c_order_array =
					true);

	template<typename T, typename ...Others>
	static MPIDataType create(Others && ... others)
	{
		return create(DataType::create<T>(), std::forward<Others>(others)...);
	}

	MPI_Datatype const & type(...) const
	{
		return type_;
	}

private:
	MPI_Datatype type_ = MPI_DATATYPE_NULL;

	bool is_commited_ = false;

};

}  // namespace simpla

#endif /* MPI_DATATYPE_H_ */
