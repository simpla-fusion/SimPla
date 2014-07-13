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
namespace _impl
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
	else if (t_index == std::type_index(typeid(float)))
	{
		*new_type = MPI_FLOAT;
	}
	else if (t_index == std::type_index(typeid(double)))
	{
		*new_type = MPI_DOUBLE;
	}
	else if (t_index == std::type_index(typeid(long double)))
	{
		*new_type = MPI_LONG_DOUBLE;
	}
	else if (t_index == std::type_index(typeid(std::complex<double>)))
	{
		*new_type = MPI_2DOUBLE_COMPLEX;
	}
	else if (t_index == std::type_index(typeid(std::complex<float>)))
	{
		*new_type = MPI_2COMPLEX;
	}
	else
	{
		MPI_Type_contiguous(size_in_byte, MPI_CHAR, new_type);
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

}  // namespace _impl
}  // namespace simpla

