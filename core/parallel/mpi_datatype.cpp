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
	if (data_type.is_compound())
	{
		//TODO create MPI structure datatype
		WARNING << "Should create structured datatype!!" << std::endl;
		MPI_Type_contiguous(data_type.ele_size_in_byte(), MPI_BYTE,
				&res.m_type_);
		MPI_Type_commit(&res.m_type_);
		res.is_commited_ = true;

////		int MPI_Type_create_struct(
////		  int count,
////		  int array_of_blocklengths[],
////		  MPI_Aint array_of_displacements[],
////		  MPI_Datatype array_of_types[],
////		  MPI_Datatype *newtype
////		);
//		int count = 0;
//
//		std::vector<int> array_of_blocklengths;
//		std::vector<MPI_Aint> array_of_displacements;
//		std::vector<MPI_Datatype> array_of_types;
//		//		  MPI_Aint array_of_displacements[],
//		//		  MPI_Datatype array_of_types[],
//		for (auto const & item : data_type.members())
//		{
//			DataType sub_datatype;
//			std::string name;
//			int offset;
//			std::tie(sub_datatype, std::ignore, offset) = item;
//
//			MPIDataType sub_mpi_type = MPIDataType::create(sub_datatype);
//
//			array_of_blocklengths.push_back(1);
//			array_of_displacements.push_back(offset);
//			array_of_types.push_back(sub_mpi_type.type());
//
//			++count;
//		}
//
//		MPI_ERROR(MPI_Type_create_struct(count,		//
//				&array_of_blocklengths[0],		//
//				&array_of_displacements[0],		//
//				&array_of_types[0], &res.m_type_));
	}
	else if (data_type.is_same<int>())
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
	else if (data_type.is_same<long double>())
	{
		res.m_type_ = MPI_LONG_DOUBLE;
	}
	else if (data_type.is_same<std::complex<double>>())
	{
		res.m_type_ = MPI_2DOUBLE_COMPLEX;
	}
	else if (data_type.is_same<std::complex<float>>())
	{
		res.m_type_ = MPI_2COMPLEX;
	}
	else
	{
		RUNTIME_ERROR("Cannot create MPI datatype:" + data_type.name());
	}

	if (data_type.is_array())
	{
		int dims[data_type.rank()];

		for (int i = 0; i < data_type.rank(); ++i)
		{
			dims[i] = data_type.extent(i);
		}
//		MPI_Datatype res2 = res;
		UNIMPLEMENTED;
//		H5_ERROR(res2 = H5Tarray_create(res, d_type.rank(), dims));
//
//		if (H5Tcommitted(res) > 0)
//			H5_ERROR(H5Tclose(res));
//
//		res = res2;
	}
	return std::move(res);
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

	return std::move(res);
}

size_t MPIDataType::size() const
{
	int s = 0;
	MPI_ERROR(MPI_Type_size(m_type_, &s));
	return s;
}
}  // namespace simpla

