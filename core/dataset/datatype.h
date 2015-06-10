/**
 * @file datatype.h
 *
 *  created on: 2014-6-2
 *      Author: salmon
 */

#ifndef DATA_TYPE_H_
#define DATA_TYPE_H_

#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <typeindex>
#include <vector>

namespace simpla
{
/**
 *  @ingroup data_interface
 *
 *  \brief  Description of data type
 *
 *  @todo this class should meet the requirement of XDR
 *  http://en.wikipedia.org/wiki/External_Data_Representation#XDR_data_types
 *  see   eXternal Data Representation Standard: Protocol Specification
 *        eXternal Data Representation: Sun Technical Notes
 *        XDR: External Data Representation Standard, RFC 1014, Sun Microsystems, Inc., USC-ISI.
 *        doc/reference/xdr/
 *
 */
struct DataType
{
	DataType();

	DataType(std::type_index t_index, size_t ele_size_in_byte,
			unsigned int ndims = 0, size_t const* dims = nullptr,
			std::string name = "");

	DataType(const DataType & other);

	~DataType();

	DataType& operator=(DataType const& other);

	void swap(DataType &);

	bool is_valid() const;

	std::string name() const;

	size_t size() const;

	size_t size_in_byte() const;

	size_t ele_size_in_byte() const;

	size_t rank() const;

	DataType element_type() const;

	size_t extent(size_t n) const;

	void extent(size_t *d) const;

	void extent(size_t rank, size_t const*d);

	std::vector<size_t> const&extents() const;

	bool is_compound() const;

	bool is_array() const;

	bool is_opaque() const;

	bool is_same(std::type_index const & other) const;

	template<typename T>
	bool is_same() const
	{
		return is_same(std::type_index(typeid(T)));
	}

	void push_back(DataType && dtype, std::string const & name, int pos = -1);

	std::vector<std::tuple<DataType, std::string, int>> const & members() const;

private:
	struct pimpl_s;
	std::unique_ptr<pimpl_s> pimpl_;

};

namespace traits
{

std::ostream & print(std::ostream & os, DataType const &self);

template<typename T>
struct datatype
{
	static DataType create(std::string const & name = "")
	{
		return DataType(std::type_index(typeid(T)), sizeof(T) / sizeof(char), 0,
				nullptr, name);
	}

};
//template<typename T>
//DataType create_opaque_datatype(std::string const & name = "")
//{
//
//	typedef typename std::remove_cv<T>::type obj_type;
//
//	typedef typename element_type<obj_type>::type value_type;
//
//	size_t ele_size_in_byte = sizeof(value_type) / sizeof(char);
//
//	return std::move(
//
//	DataType(std::type_index(typeid(value_type)),
//
//	ele_size_in_byte,
//
//	rank<obj_type>::value,
//
//	&dimensions<obj_type>::value[0],
//
//	name)
//
//	);
//
//}
}// namespace traits

}
// namespace simpla

#endif /* DATA_TYPE_H_ */
