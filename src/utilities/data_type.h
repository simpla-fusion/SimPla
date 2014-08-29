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
#include "../utilities/log.h"
namespace simpla
{
/**
 *  \brief  Desciption of data type
 *
 */
struct DataType
{
public:
	DataType()
			: t_index_(std::type_index(typeid(void)))
	{
	}

	DataType(std::type_index t_index, size_t ele_size_in_byte, unsigned int ndims = 0, size_t* dims = nullptr)
			: t_index_(t_index), ele_size_in_byte_(ele_size_in_byte), ndims(ndims)
	{
		if (ndims > 0 && dims != nullptr)
		{
			for (int i = 0; i < ndims; ++i)
			{
				dimensions_[i] = dims[i];
			}

		}
	}
	DataType(const DataType & other)
			: ele_size_in_byte_(other.ele_size_in_byte_), t_index_(other.t_index_), ndims(other.ndims)
	{
		for (int i = 0; i < ndims; ++i)
		{
			dimensions_[i] = other.dimensions_[i];
		}

		std::copy(other.data.begin(), other.data.end(), std::back_inserter(data));
	}

	~DataType()
	{
	}

	DataType& operator=(DataType const& other)
	{
		t_index_ = other.t_index_;
		ele_size_in_byte_ = other.ele_size_in_byte_;
		ndims = other.ndims;
		for (int i = 0; i < ndims; ++i)
		{
			dimensions_[i] = other.dimensions_[i];
		}
		std::copy(other.data.begin(), other.data.end(), std::back_inserter(data));
		return *this;
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

		for (int i = 0; i < ndims; ++i)
		{
			res *= dimensions_[i];
		}
		return res;
	}

	bool is_compound() const
	{
		return data.size() > 0;
	}
	template<typename T> bool is_same() const
	{
		return t_index_ == std::type_index(typeid(T));
	}

	template<typename T>
	void push_back(std::string const & name, int pos = -1)
	{
		if (pos < 0)
		{
			if (data.empty())
			{
				pos = 0;
			}
			else
			{
				pos = std::get<2>(*(data.rbegin())) + std::get<0>(*(data.rbegin())).size_in_byte();
			}
		}

		data.push_back(std::make_tuple(DataType::create<T>(), name, pos));

	}

	size_t ele_size_in_byte_ = 0;
	std::type_index t_index_;
	unsigned int ndims = 0;
	size_t dimensions_[MAX_NDIMS_OF_ARRAY];

	std::vector<std::tuple<DataType, std::string, int>> data;

};

}
// namespace simpla

#endif /* DATA_TYPE_H_ */
