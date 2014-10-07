/**
 * mpi_datatype.cpp
 *
 * \date 2014年7月8日
 * \author salmon
 */
#include <mpi.h>
#include <typeinfo>
#include <typeindex>
#include "../utilities/data_type.h"
#include "mpi_datatype.h"

namespace simpla
{

MPIDataType MPIDataType::create(DataType const & data_type)
{
	MPIDataType res;

	res.is_commited_ = false;

	if (data_type.t_index_ == std::type_index(typeid(int)))
	{
		res.type_ = MPI_INT;
	}
	else if (data_type.t_index_ == std::type_index(typeid(long)))
	{
		res.type_ = MPI_LONG;
	}
	else if (data_type.t_index_ == std::type_index(typeid(unsigned long)))
	{
		res.type_ = MPI_UNSIGNED_LONG;
	}
	else if (data_type.t_index_ == std::type_index(typeid(float)))
	{
		res.type_ = MPI_FLOAT;
	}
	else if (data_type.t_index_ == std::type_index(typeid(double)))
	{
		res.type_ = MPI_DOUBLE;
	}
//	else if (data_type.t_index_ == std::type_index(typeid(long double)))
//	{
//		res.type_ = MPI_LONG_DOUBLE;
//	}
//	else if (data_type.t_index_ == std::type_index(typeid(std::complex<double>)))
//	{
//		res.type_ = MPI_2DOUBLE_COMPLEX;
//	}
//	else if (data_type.t_index_ == std::type_index(typeid(std::complex<float>)))
//	{
//		res.type_ = MPI_2COMPLEX;
//	}
	else
	{
		MPI_Type_contiguous(data_type.ele_size_in_byte_, MPI_BYTE, &res.type_);
		MPI_Type_commit(&res.type_);
		res.is_commited_ = true;
	}
	return (res);
}

MPIDataType MPIDataType::create(DataType const & data_type, unsigned int NDIMS, size_t const *outer,
        size_t const * inner, size_t const * start, bool c_order_array)
{
	MPIDataType res;

	const int v_ndims = data_type.ndims;

	int outer1[NDIMS + v_ndims];
	int inner1[NDIMS + v_ndims];
	int start1[NDIMS + v_ndims];
	for (int i = 0; i < NDIMS; ++i)
	{
		outer1[i] = outer[i];
		inner1[i] = inner[i];
		start1[i] = start[i];
	}

	for (int i = 0; i < v_ndims; ++i)
	{
		outer1[NDIMS + i] = data_type.dimensions_[i];
		inner1[NDIMS + i] = data_type.dimensions_[i];
		start1[NDIMS + i] = 0;
	}

	auto ele_type = MPIDataType::create(data_type);

	MPI_Type_create_subarray(NDIMS + v_ndims, outer1, inner1, start1, (c_order_array ? MPI_ORDER_C : MPI_ORDER_FORTRAN),
	        ele_type.type(), &res.type_);

	MPI_Type_commit(&res.type_);

	res.is_commited_ = true;

	return res;
}
}  // namespace simpla

