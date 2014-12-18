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
 *  \brief  Description of data type
 *
 */
struct DataType
{
	DataType();

	DataType(std::type_index t_index, size_t ele_size_in_byte,
			unsigned int ndims = 0, size_t* dims = nullptr, std::string name =
					"");
	DataType(const DataType & other);
	DataType(DataType && other);
	~DataType();

	DataType& operator=(DataType const& other);

	bool is_valid() const;

	template<typename T>
	static DataType create(std::string const &name = "")
	{
		typedef typename std::remove_cv<T>::type type;
		static_assert( nTuple_traits<type>::ndims< MAX_NDIMS_OF_ARRAY,
				"the NDIMS of nTuple is bigger than MAX_NDIMS_OF_ARRAY");

		typedef typename nTuple_traits<type>::value_type value_type;

		size_t ele_size_in_byte = sizeof(value_type) / sizeof(ByteType);

		auto ndims = nTuple_traits<type>::dimensions::size();

		auto dimensions = seq2ntuple(
				typename nTuple_traits<type>::dimensions());

		return std::move(
				DataType(std::type_index(typeid(value_type)), ele_size_in_byte,
						ndims, &dimensions[0], name));
	}

	size_t size_in_byte() const;

	size_t ele_size_in_byte() const;

	size_t rank() const;

	size_t extent(size_t n) const;

	bool is_compound() const;

	bool is_same(std::type_index const & other) const;

	template<typename T>
	bool is_same() const
	{
		return is_same(std::type_index(typeid(T)));
	}

	void push_back(DataType const & dtype, std::string const & name, int pos =
			-1);

	template<typename T>
	void push_back(std::string const & name, int pos = -1)
	{
		push_back(DataType::create<T>(), name, pos);
	}

	std::ostream & print(std::ostream & os) const;

	std::vector<std::tuple<DataType, std::string, int>> const & members() const;

private:
	struct pimpl_s;
	pimpl_s *pimpl_;

};
HAS_STATIC_MEMBER_FUNCTION(datatype)
template<typename T>
auto make_datatype(
		std::string const & name = "")
				ENABLE_IF_DECL_RET_TYPE((!has_static_member_function_datatype<T>::value),DataType::create<T>(name))
template<typename T>
auto make_datatype(
		std::string const & name = "")
				ENABLE_IF_DECL_RET_TYPE((has_static_member_function_datatype<T>::value),T::datatype())

}
// namespace simpla

#endif /* DATA_TYPE_H_ */
