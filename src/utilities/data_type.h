/*
 * data_type.h
 *
 *  created on: 2014-6-2
 *      Author: salmon
 */

#ifndef DATA_TYPE_H_
#define DATA_TYPE_H_
#include <typeinfo>

#include "../utilities/primitives.h"
#include "../utilities/ntuple.h"

namespace simpla
{
/**
 *  @todo need support for compound data type
 */
struct DataType
{
public:

	DataType(std::type_info const & t_info, size_t ele_size_in_byte, unsigned int ndims = 0, size_t* dims = nullptr)
			: t_info_(t_info), ele_size_in_byte_(ele_size_in_byte), NDIMS(ndims)
	{
		if (ndims > 0 && dims != nullptr)
		{
			for (int i = 0; i < NDIMS; ++i)
			{
				dimensions_[i] = dims[i];
			}

		}
	}

	~DataType()
	{
	}

	template<typename T> static DataType create()
	{
		static_assert( nTupleTraits<T>::NDIMS< MAX_NDIMS_OF_ARRAY,"the NDIMS of ntuple is bigger than MAX_NDIMS_OF_ARRAY");

		typedef typename nTupleTraits<T>::element_type element_type;

		size_t ele_size_in_byte = sizeof(element_type) / sizeof(ByteType);

		unsigned int NDIMS = nTupleTraits<T>::NDIMS;

		size_t dimensions[nTupleTraits<T>::NDIMS + 1];

		nTupleTraits<T>::get_dimensions(dimensions);

		return std::move(DataType(typeid(element_type), ele_size_in_byte, NDIMS, dimensions));
	}

	size_t size_in_byte() const
	{
		size_t res = ele_size_in_byte_;

		for (int i = 0; i < NDIMS; ++i)
		{
			res *= dimensions_[i];
		}
		return res;
	}

	const size_t ele_size_in_byte_;
	const std::type_info & t_info_;
	const unsigned int NDIMS = 0;
	size_t dimensions_[MAX_NDIMS_OF_ARRAY];

};

}
// namespace simpla

#endif /* DATA_TYPE_H_ */
