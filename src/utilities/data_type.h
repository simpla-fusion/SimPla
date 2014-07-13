/*
 * data_type.h
 *
 *  created on: 2014-6-2
 *      Author: salmon
 */

#ifndef DATA_TYPE_H_
#define DATA_TYPE_H_
#include <typeinfo>
#include <typeindex>
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
	DataType()
			: t_index_(std::type_index(typeid(void)))
	{

	}

	DataType(std::type_index t_index, size_t ele_size_in_byte, unsigned int ndims = 0, size_t* dims = nullptr)
			: t_index_(t_index), ele_size_in_byte_(ele_size_in_byte), NDIMS(ndims)
	{
		if (ndims > 0 && dims != nullptr)
		{
			for (int i = 0; i < NDIMS; ++i)
			{
				dimensions_[i] = dims[i];
			}

		}
	}
	DataType(const DataType & other)
			: ele_size_in_byte_(other.ele_size_in_byte_), t_index_(other.t_index_), NDIMS(other.NDIMS)
	{
		for (int i = 0; i < NDIMS; ++i)
		{
			dimensions_[i] = other.dimensions_[i];
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

		return std::move(DataType(std::type_index(typeid(element_type)), ele_size_in_byte, NDIMS, dimensions));
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

	size_t ele_size_in_byte_ = 0;
	std::type_index t_index_;
	unsigned int NDIMS = 0;
	size_t dimensions_[MAX_NDIMS_OF_ARRAY];

};

}
// namespace simpla

#endif /* DATA_TYPE_H_ */
