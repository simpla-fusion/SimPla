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

namespace simpla
{

bool GetMPIType(std::type_index const & t_index, size_t size_in_byte, MPI_Datatype * new_type)
{
	bool is_commited = false;

	if (t_index == std::type_index(typeid(int)))
	{
		*new_type = MPI_INT;
	}
	else if (t_index == std::type_index(typeid(long)))
	{
		*new_type = MPI_LONG;
	}
	else if (t_index == std::type_index(typeid(unsigned long)))
	{
		*new_type = MPI_UNSIGNED_LONG;
	}
	else if (t_index == std::type_index(typeid(float)))
	{
		*new_type = MPI_FLOAT;
	}
	else if (t_index == std::type_index(typeid(double)))
	{
		*new_type = MPI_DOUBLE;
	}
//	else if (t_index == std::type_index(typeid(long double)))
//	{
//		*new_type = MPI_LONG_DOUBLE;
//	}
//	else if (t_index == std::type_index(typeid(std::complex<double>)))
//	{
//		*new_type = MPI_2DOUBLE_COMPLEX;
//	}
//	else if (t_index == std::type_index(typeid(std::complex<float>)))
//	{
//		*new_type = MPI_2COMPLEX;
//	}
	else
	{
		MPI_Type_contiguous(size_in_byte, MPI_BYTE, new_type);
		MPI_Type_commit(new_type);
		is_commited = true;
	}
	return is_commited;
}

bool GetMPIType(DataType const & datatype_desc, MPI_Datatype * new_type)
{
	bool is_commited = false;

	if (datatype_desc.NDIMS == 0)
	{
		is_commited = GetMPIType(datatype_desc.t_index_, datatype_desc.ele_size_in_byte_, new_type);
	}
	else
	{
		int ndims = datatype_desc.NDIMS;

		int dims[ndims];

		for (int i = 0; i < ndims; ++i)
		{
			dims[i] = datatype_desc.dimensions_[i];
		}

		MPI_Datatype ele_type;

		GetMPIType(datatype_desc.t_index_, datatype_desc.ele_size_in_byte_, &ele_type);

		MPI_Type_contiguous(ndims, ele_type, new_type);

		MPI_Type_commit(new_type);

		is_commited = true;
	}

	return is_commited;
}

template<typename T, unsigned int NDIMS, typename TI>
static MPIDataType create(nTuple<NDIMS, TI> const &outer, nTuple<NDIMS, TI> const &inner,
        nTuple<NDIMS, TI> const &start, unsigned int array_order_ =
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


}  // namespace simpla

