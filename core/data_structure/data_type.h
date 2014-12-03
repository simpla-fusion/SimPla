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
 *  \page DataTypeConcept Data type concept
 *
 *   TODO add sth for datatype
 *
 *  \brief  Desciption of data type
 *
 */
struct DataType
{
public:
	DataType() :
			t_index_(std::type_index(typeid(void)))
	{
	}

	DataType(std::type_index t_index, size_t ele_size_in_byte,
			unsigned int ndims = 0, size_t* dims = nullptr) :
			t_index_(t_index), ele_size_in_byte_(ele_size_in_byte), ndims(ndims)
	{
		if (ndims > 0 && dims != nullptr)
		{
			for (int i = 0; i < ndims; ++i)
			{
				dimensions_[i] = dims[i];
			}

		}
	}

	DataType(const DataType & other) :
			ele_size_in_byte_(other.ele_size_in_byte_), t_index_(
					other.t_index_), ndims(other.ndims)
	{

		dimensions_ = other.dimensions_;

		std::copy(other.data.begin(), other.data.end(),
				std::back_inserter(data));
	}

	~DataType()
	{
	}

	DataType& operator=(DataType const& other)
	{
		t_index_ = other.t_index_;
		ele_size_in_byte_ = other.ele_size_in_byte_;
		ndims = other.ndims;
		dimensions_ = other.dimensions_;

		std::copy(other.data.begin(), other.data.end(),
				std::back_inserter(data));
		return *this;
	}

	template<typename T> static DataType create()
	{
		static_assert( nTuple_traits<T>::ndims< MAX_NDIMS_OF_ARRAY,
				"the NDIMS of nTuple is bigger than MAX_NDIMS_OF_ARRAY");

		typedef typename nTuple_traits<T>::value_type value_type;

		size_t ele_size_in_byte = sizeof(value_type) / sizeof(ByteType);

		auto ndims = nTuple_traits<T>::dimensions::size();

		auto dimensions = seq2ntuple(typename nTuple_traits<T>::dimensions());

		return std::move(
				DataType(std::type_index(typeid(value_type)), ele_size_in_byte,
						ndims, &dimensions[0]));
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
				pos = std::get<2>(*(data.rbegin()))
						+ std::get<0>(*(data.rbegin())).size_in_byte();
			}
		}

		data.push_back(std::make_tuple(DataType::create<T>(), name, pos));

	}

	size_t ele_size_in_byte_ = 0;
	std::type_index t_index_;
	size_t ndims = 0;
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> dimensions_;

	std::vector<std::tuple<DataType, std::string, int>> data;

};
HAS_STATIC_MEMBER_FUNCTION(data_desc)
template<typename T>
auto make_datatype()
		ENABLE_IF_DECL_RET_TYPE((!has_static_member_function_data_desc<T>::value),DataType::create<T>())
template<typename T>
auto make_datatype()
		ENABLE_IF_DECL_RET_TYPE((has_static_member_function_data_desc<T>::value),T::data_desc())

}
// namespace simpla

#endif /* DATA_TYPE_H_ */
