/**
 * mpi_datatype.cpp
 *
 * \date 2014年7月8日
 * \author salmon
 */
#include <mpi.h>
#include <typeinfo>
#include <typeindex>
#include "mpi_datatype.h"
#include "../data_structure/data_type.h"

namespace simpla
{

MPIDataType::MPIDataType()
{
}

MPIDataType::~MPIDataType()
{
	if (is_commited_)
		MPI_Type_free(&type_);
}

MPIDataType MPIDataType::create(DataType const & data_type)
{
	MPIDataType res;

	res.is_commited_ = false;

	if (data_type.is_same<int>())
	{
		res.type_ = MPI_INT;
	}
	else if (data_type.is_same<int>())
	{
		res.type_ = MPI_LONG;
	}
	else if (data_type.is_same<unsigned long>())
	{
		res.type_ = MPI_UNSIGNED_LONG;
	}
	else if (data_type.is_same<float>())
	{
		res.type_ = MPI_FLOAT;
	}
	else if (data_type.is_same<double>())
	{
		res.type_ = MPI_DOUBLE;
	}
//	else if (data_type.is_same<long double>())
//	{
//		res.type_ = MPI_LONG_DOUBLE;
//	}
//	else if (data_type.is_same<std::complex<double>>())
//	{
//		res.type_ = MPI_2DOUBLE_COMPLEX;
//	}
//	else if (data_type.is_same<std::complex<float>>())
//	{
//		res.type_ = MPI_2COMPLEX;
//	}
	else
	{
		MPI_Type_contiguous(data_type.ele_size_in_byte(), MPI_BYTE, &res.type_);
		MPI_Type_commit(&res.type_);
		res.is_commited_ = true;
	}
	return (res);
}

MPIDataType MPIDataType::create(DataType const & data_type, unsigned int ndims,
		size_t const * dims,        //
		size_t const * offset,      //
		size_t const * stride,      //
		size_t const * count,       //
		size_t const * block,       //
		bool c_order_array)
{

	MPIDataType res;

	unsigned int mdims = ndims + data_type.rank();

	nTuple<int, MAX_NDIMS_OF_ARRAY> m_dims;
	nTuple<int, MAX_NDIMS_OF_ARRAY> m_count;
	nTuple<int, MAX_NDIMS_OF_ARRAY> m_stride;
	nTuple<int, MAX_NDIMS_OF_ARRAY> m_offset;
	nTuple<int, MAX_NDIMS_OF_ARRAY> m_block;

	auto old_type = MPIDataType::create(data_type);

	if (dims == nullptr)
	{
		WARNING << "Undefined array dimensions!!";

		return old_type;
	}
	else
	{
		m_dims = dims;
	}
	if (offset == nullptr)
	{
		m_offset = 0;
	}
	else
	{
		m_offset = offset;
	}

	if (count == nullptr)
	{
		m_count = m_dims;
	}
	else
	{
		m_count = count;
	}

	if (stride != nullptr || block != nullptr)
	{
		//TODO create mpi datatype with stride and block
		WARNING << "UNIMPLEMENTED!! 'stride'  and 'block' are ignored! "
				<< std::endl;
	}

	for (int i = 0; i < data_type.rank(); ++i)
	{
		m_dims[ndims + i] = data_type.extent(i);
		m_count[ndims + i] = data_type.extent(i);
		m_offset[ndims + i] = 0;
	}

	MPI_Type_create_subarray(ndims + data_type.rank(), &m_dims[0], &m_count[0],
			&m_offset[0], (c_order_array ? MPI_ORDER_C : MPI_ORDER_FORTRAN),
			old_type.type(), &res.type_);

	MPI_Type_commit(&res.type_);

	res.is_commited_ = true;

	return res;
}
}  // namespace simpla

