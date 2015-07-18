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
#include "utilities.h"
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

MPIDataType MPIDataType::create(DataType const & data_type, //
		unsigned int ndims, //
		size_t const * p_dims,        //
		size_t const * p_offset,      //
		size_t const * p_stride,      //
		size_t const * p_count,       //
		size_t const * p_block,       //
		bool c_order_array)
{

	MPI_Datatype res_type;

	bool is_predefined = true;

	if (data_type.is_compound())
	{
		is_predefined = false;
		//TODO create MPI structure datatype
		//		WARNING << "TODO: create structured datatype!!" << std::endl;
		//		MPI_Type_contiguous(data_type.ele_size_in_byte(), MPI_BYTE,
		//				&res.m_type_);

		////		int MPI_Type_create_struct(
		////		  int count,
		////		  int array_of_blocklengths[],
		////		  MPI_Aint array_of_displacements[],
		////		  MPI_Datatype array_of_types[],
		////		  MPI_Datatype *newtype
		////		);
		std::vector<MPIDataType> dtypes;
		std::vector<int> array_of_blocklengths;
		std::vector<MPI_Aint> array_of_displacements;
		std::vector<MPI_Datatype> array_of_types;
		//		  MPI_Aint array_of_displacements[],
		//		  MPI_Datatype array_of_types[],
		for (auto const & item : data_type.members())
		{
			DataType sub_datatype;

			int offset;

			std::tie(sub_datatype, std::ignore, offset) = item;

			int block_length = 1;

			for (int i = 0; i < sub_datatype.rank(); ++i)
			{
				block_length *= sub_datatype.extent(i);
			}

			dtypes.push_back(MPIDataType::create(sub_datatype.element_type()));

			array_of_blocklengths.push_back(block_length);
			array_of_displacements.push_back(offset);
			array_of_types.push_back(dtypes.rbegin()->type());

		}

		MPI_ERROR(MPI_Type_create_struct(		//
				array_of_blocklengths.size(),		//
				&array_of_blocklengths[0],		//
				&array_of_displacements[0],		//
				&array_of_types[0], &res_type));

	}
	else if (data_type.is_same<int>())
	{
		res_type = MPI_INT;
	}
	else if (data_type.is_same<long>())
	{
		res_type = MPI_LONG;
	}
	else if (data_type.is_same<unsigned int>())
	{
		res_type = MPI_UNSIGNED;
	}
	else if (data_type.is_same<unsigned long>())
	{
		res_type = MPI_UNSIGNED_LONG;
	}
	else if (data_type.is_same<float>())
	{
		res_type = MPI_FLOAT;
	}
	else if (data_type.is_same<double>())
	{
		res_type = MPI_DOUBLE;
	}
	else if (data_type.is_same<long double>())
	{
		res_type = MPI_LONG_DOUBLE;
	}
	else if (data_type.is_same<std::complex<double>>())
	{
		res_type = MPI_2DOUBLE_COMPLEX;
	}
	else if (data_type.is_same<std::complex<float>>())
	{
		res_type = MPI_2COMPLEX;
	}
	else
	{
		RUNTIME_ERROR("Cannot create MPI datatype:" + data_type.name());
	}

	if (data_type.is_array() || (ndims > 0 && p_dims != nullptr))
	{

		unsigned int mdims = ndims + data_type.rank();

		nTuple<int, MAX_NDIMS_OF_ARRAY> l_dims;
		nTuple<int, MAX_NDIMS_OF_ARRAY> l_offset;
		nTuple<int, MAX_NDIMS_OF_ARRAY> l_stride;
		nTuple<int, MAX_NDIMS_OF_ARRAY> l_count;
		nTuple<int, MAX_NDIMS_OF_ARRAY> l_block;

		auto old_type = MPIDataType::create(data_type);

		if (p_dims != nullptr)
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

		MPI_Datatype ele_type = res_type;

		MPI_ERROR(MPI_Type_create_subarray(ndims + data_type.rank(), //
				&l_dims[0], &l_count[0], &l_offset[0],//
				(c_order_array ? MPI_ORDER_C : MPI_ORDER_FORTRAN),//
				ele_type, &res_type));

		if (!is_predefined)
		{
			MPI_Type_free(&ele_type);
		}
		is_predefined = false;
	}

	if (!is_predefined)
	{
		MPI_Type_commit(&res_type);
	}

	MPIDataType res;
	res.m_type_ = res_type;
	res.is_commited_ = !is_predefined;
//	CHECK(data_type.size_in_byte());
//	CHECK(res.size());
	return std::move(res);
}

size_t MPIDataType::size() const
{
	int s = 0;
	MPI_ERROR(MPI_Type_size(m_type_, &s));
	return s;
}
}  // namespace simpla

