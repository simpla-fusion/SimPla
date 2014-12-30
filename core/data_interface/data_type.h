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
 *  @ingroup data_interface
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

	void swap(DataType &);

	bool is_valid() const;

	std::string name() const;

	size_t size() const;

	size_t size_in_byte() const;

	size_t ele_size_in_byte() const;

	size_t rank() const;

	size_t extent(size_t n) const;

	void extent(size_t rank, size_t const*d);

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

	std::ostream & print(std::ostream & os) const;

	std::vector<std::tuple<DataType, std::string, int>> const & members() const;

	template<typename T>
	static DataType create_opaque_type(std::string const & name = "")
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

private:
	struct pimpl_s;
	pimpl_s *pimpl_;

	HAS_STATIC_MEMBER_FUNCTION(datatype);

	template <typename T>
	static DataType create_(std::false_type, std::string const & name = "")
	{
		return std::move(create_opaque_type<T>(name));
	}

	template<typename T>
	static DataType create_(std::true_type,std::string const & name = "")
	{
		return std::move(T::datatype());
	}

public:
	template<typename T>
	static DataType create( std::string const & name = "")
	{
		return std::move(create_<T>(std::integral_constant<bool,
						has_static_member_function_datatype<T>::value>(),name));
	}

}
;

/**
 * @ingroup data_interface
 * @{
 * create datatype
 * @param name
 * @return
 */
template<typename T>
auto make_datatype(std::string const & name = "")
DECL_RET_TYPE( DataType::create<T>(name))
/**@}  */
}
// namespace simpla

#endif /* DATA_TYPE_H_ */
