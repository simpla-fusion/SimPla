/*
 * mpi_datatype.h
 *
 *  created on: 2014-5-26
 *      Author: salmon
 */

#ifndef MPI_DATATYPE_H_
#define MPI_DATATYPE_H_

#include <mpi.h>
#include <stddef.h>
#include <cstdbool>

#include "../dataset/datatype.h"

namespace simpla
{
namespace _impl
{

bool GetMPIType(DataType const & datatype_desc, MPI_Datatype * new_type);

}  // namespace _impl

/**
 *  @ingroup MPI
 *  \brief MPI convert C++ data type and data space to mpi data type
 */
struct MPIDataType
{

	MPIDataType();

	MPIDataType(MPIDataType const &);

	~MPIDataType();

//	static MPIDataType create(DataType const &);

	static MPIDataType create(DataType const & data_type,
			unsigned int ndims = 0, size_t const * dims = nullptr,
			size_t const * count = nullptr, size_t const * offset = nullptr,
			size_t const * stride = nullptr, size_t const * block = nullptr,
			bool c_order_array = true);

	template<typename T, typename ...Others>
	static MPIDataType create(Others && ... others)
	{
		return create(traits::datatype<T>::create(), std::forward<Others>(others)...);
	}

	MPI_Datatype const & type(...) const
	{
		return m_type_;
	}

	size_t size() const;

private:

	MPI_Datatype m_type_ = MPI_DATATYPE_NULL;

	bool is_commited_ = false;

};

}  // namespace simpla

#endif /* MPI_DATATYPE_H_ */
