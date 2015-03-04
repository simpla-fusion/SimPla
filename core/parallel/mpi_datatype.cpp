/**
 * mpi_datatype.cpp
 *
 * \date 2014年7月8日
 * \author salmon
 */
#include <mpi.h>
#include <typeinfo>
#include <typeindex>
#include "mpi_comm.h"
#include "mpi_datatype.h"
#include "../utilities/utilities.h"
namespace simpla
{

MPIDataType::MPIDataType()
{
}
MPIDataType::MPIDataType(MPIDataType const & other)
{
	MPI_ERROR(MPI_Type_dup(other.type(), &m_type_));
}

MPIDataType::~MPIDataType()
{
	if (is_commited_)
		MPI_Type_free(&m_type_);
}

MPIDataType MPIDataType::create(DataType const & data_type)
{
	MPIDataType res;

	res.is_commited_ = false;

	if (data_type.is_same<int>())
	{
		res.m_type_ = MPI_INT;
	}
	else if (data_type.is_same<int>())
	{
		res.m_type_ = MPI_LONG;
	}
	else if (data_type.is_same<unsigned long>())
	{
		res.m_type_ = MPI_UNSIGNED_LONG;
	}
	else if (data_type.is_same<float>())
	{
		res.m_type_ = MPI_FLOAT;
	}
	else if (data_type.is_same<double>())
	{
		res.m_type_ = MPI_DOUBLE;
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
		MPI_Type_contiguous(data_type.ele_size_in_byte(), MPI_BYTE, &res.m_type_);
		MPI_Type_commit(&res.m_type_);
		res.is_commited_ = true;
	}
	return (res);
}

MPIDataType MPIDataType::create(DataType const & data_type, unsigned int ndims,
		size_t const * p_dims,        //
		size_t const * p_offset,      //
		size_t const * p_stride,      //
		size_t const * p_count,       //
		size_t const * p_block,       //
		bool c_order_array)
{

	MPIDataType res;

	unsigned int mdims = ndims + data_type.rank();

	nTuple<int, MAX_NDIMS_OF_ARRAY> l_dims;
	nTuple<int, MAX_NDIMS_OF_ARRAY> l_offset;
	nTuple<int, MAX_NDIMS_OF_ARRAY> l_stride;
	nTuple<int, MAX_NDIMS_OF_ARRAY> l_count;
	nTuple<int, MAX_NDIMS_OF_ARRAY> l_block;

	auto old_type = MPIDataType::create(data_type);

	if (p_dims == nullptr)
	{
		WARNING << "Undefined array dimensions!!";

		return old_type;
	}
	else
	{
		l_dims = p_dims;
	}
	if (p_offset == nullptr)
	{
		l_offset = 0;
	}
	else
	{
		l_offset = p_offset;
	}

	if (p_count == nullptr)
	{
		l_count = l_dims;
	}
	else
	{
		l_count = p_count;
	}

	if (p_stride != nullptr || p_block != nullptr)
	{
		//TODO create mpi datatype with stride and block
		WARNING << "UNIMPLEMENTED!! 'stride'  and 'block' are ignored! "
				<< std::endl;
	}

	for (int i = 0; i < data_type.rank(); ++i)
	{
		l_dims[ndims + i] = data_type.extent(i);
		l_count[ndims + i] = data_type.extent(i);
		l_offset[ndims + i] = 0;
	}

	int mpi_error = MPI_Type_create_subarray(ndims + data_type.rank(),
			&l_dims[0], &l_count[0], &l_offset[0],
			(c_order_array ? MPI_ORDER_C : MPI_ORDER_FORTRAN), old_type.type(),
			&res.m_type_);

	MPI_Type_commit(&res.m_type_);

	res.is_commited_ = true;

	return res;
}
}  // namespace simpla

