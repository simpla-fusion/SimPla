/**
 * @file datatype.h
 *
 *  created on: 2014-6-2
 *      Author: salmon
 */

#ifndef DATA_TYPE_H_
#define DATA_TYPE_H_

#include <stddef.h>
#include <cstdbool>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <vector>

#include "../gtl/type_traits.h"

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

	bool is_compound() const;bool is_array() const;bool is_opaque() const;bool is_same(
			std::type_index const & other) const;
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

		typedef typename traits::value_type<type>::type value_type;

		size_t ele_size_in_byte = sizeof(value_type) / sizeof(char);

		return std::move(

		DataType(std::type_index(typeid(value_type)),

		ele_size_in_byte,

		traits::rank<type>::value,

		&traits::dimensions<type>::value[0],

		name)

		);
	}

private:
	struct pimpl_s;
	std::unique_ptr<pimpl_s> pimpl_;

	HAS_STATIC_MEMBER_FUNCTION (datatype);

	template<typename T>
	static DataType create_(std::false_type, std::string const & name = "")
	{
		return std::move(create_opaque_type<T>(name));
	}

	template<typename T>
	static DataType create_(std::true_type, std::string const & name = "")
	{
		return std::move(T::datatype());
	}

public:
	template<typename T>
	static DataType create(std::string const & name = "")
	{
		return std::move(
				create_<T>(
						std::integral_constant<bool,
						has_static_member_function_datatype<T>::value>(),
						name));
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
DECL_RET_TYPE (DataType::create<T>(name))
/**@}  */

/*
 * Count the number of arguments passed to MACRO, very carefully
 * tiptoeing around an MSVC bug where it improperly expands __VA_ARGS__ as a
 * single token in argument lists.  See these URLs for details:
 *
 *   http://stackoverflow.com/questions/9183993/msvc-variadic-macro-expansion/9338429#9338429
 *   http://connect.microsoft.com/VisualStudio/feedback/details/380090/variadic-macro-replacement
 *   http://cplusplus.co.il/2010/07/17/variadic-macro-to-count-number-of-arguments/#comment-644
 */
#define COUNT_MACRO_ARGS_IMPL2(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18, count, ...) count
#define COUNT_MACRO_ARGS_IMPL(args) COUNT_MACRO_ARGS_IMPL2 args
#define COUNT_MACRO_ARGS(...) COUNT_MACRO_ARGS_IMPL((__VA_ARGS__,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0))

/* Pick the right helper macro to invoke. */
#define SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_)  typename array_to_ntuple_convert<_T0_>::type _N0_;

#define SP_DEFINE_STRUCT_MEMBER_HELPER4(_T0_,_N0_,_T1_,_N1_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_DEFINE_STRUCT_MEMBER_HELPER2(_T1_,_N1_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER6(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_DEFINE_STRUCT_MEMBER_HELPER4(_T1_,_N1_,_T2_,_N2_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER8(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_DEFINE_STRUCT_MEMBER_HELPER6(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER10(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_DEFINE_STRUCT_MEMBER_HELPER8(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER12(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_DEFINE_STRUCT_MEMBER_HELPER10(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER14(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_DEFINE_STRUCT_MEMBER_HELPER12(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER16(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_DEFINE_STRUCT_MEMBER_HELPER14(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER18(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_DEFINE_STRUCT_MEMBER_HELPER16(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)

#define SP_DEFINE_STRUCT_MEMBER_CHOOSE_HELPER1(count) SP_DEFINE_STRUCT_MEMBER_HELPER##count
#define SP_DEFINE_STRUCT_MEMBER_CHOOSE_HELPER(count) SP_DEFINE_STRUCT_MEMBER_CHOOSE_HELPER1(count)
#define SP_DEFINE_STRUCT_MEMBER(...) SP_DEFINE_STRUCT_MEMBER_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (__VA_ARGS__)

//#define SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) _N0_
//#define SP_PARTICLE_GET_NAME_HELPER4(_T0_,_N0_,_T1_,_N1_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER2(_T1_,_N1_)
//#define SP_PARTICLE_GET_NAME_HELPER6(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_)  SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER4(_T1_,_N1_,_T2_,_N2_)
//#define SP_PARTICLE_GET_NAME_HELPER8(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER6(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
//#define SP_PARTICLE_GET_NAME_HELPER10(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER8(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
//#define SP_PARTICLE_GET_NAME_HELPER12(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER10(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
//#define SP_PARTICLE_GET_NAME_HELPER14(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER12(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
//#define SP_PARTICLE_GET_NAME_HELPER16(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER14(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
//#define SP_PARTICLE_GET_NAME_HELPER18(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER16(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)
//
//#define SP_PARTICLE_GET_NAME_CHOOSE_HELPER1(count) SP_PARTICLE_GET_NAME_HELPER##count
//#define SP_PARTICLE_GET_NAME_CHOOSE_HELPER(count) SP_PARTICLE_GET_NAME_CHOOSE_HELPER1(count)
//#define SP_PARTICLE_GET_NAME(...) SP_PARTICLE_GET_NAME_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (__VA_ARGS__)

#define SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) d_type.push_back(make_datatype<typename array_to_ntuple_convert<_T0_>::type>(), #_N0_, offsetof(_S_NAME_, _N0_));
#define SP_DEFINE_STRUCT_DESC_HELPER4(_S_NAME_,_T0_,_N0_,_T1_,_N1_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T1_,_N1_)
#define SP_DEFINE_STRUCT_DESC_HELPER6(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_)  SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_DEFINE_STRUCT_DESC_HELPER4(_S_NAME_,_T1_,_N1_,_T2_,_N2_)
#define SP_DEFINE_STRUCT_DESC_HELPER8(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_DEFINE_STRUCT_DESC_HELPER6(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
#define SP_DEFINE_STRUCT_DESC_HELPER10(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_DEFINE_STRUCT_DESC_HELPER8(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
#define SP_DEFINE_STRUCT_DESC_HELPER12(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_DEFINE_STRUCT_DESC_HELPER10(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
#define SP_DEFINE_STRUCT_DESC_HELPER14(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_DEFINE_STRUCT_DESC_HELPER12(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
#define SP_DEFINE_STRUCT_DESC_HELPER16(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_DEFINE_STRUCT_DESC_HELPER14(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
#define SP_DEFINE_STRUCT_DESC_HELPER18(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_DEFINE_STRUCT_DESC_HELPER16(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)

#define SP_DEFINE_STRUCT_DESC_CHOOSE_HELPER1(count) SP_DEFINE_STRUCT_DESC_HELPER##count
#define SP_DEFINE_STRUCT_DESC_CHOOSE_HELPER(count) SP_DEFINE_STRUCT_DESC_CHOOSE_HELPER1(count)
#define SP_DEFINE_STRUCT_DESC(_S_NAME_,...) SP_DEFINE_STRUCT_DESC_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (_S_NAME_,__VA_ARGS__)

#define SP_DEFINE_STRUCT(_S_NAME_,...)                                 \
struct _S_NAME_                                                  \
{                                                                \
	SP_DEFINE_STRUCT_MEMBER(__VA_ARGS__)                                   \
	static DataType datatype()                             \
	{                                                             \
		auto d_type = DataType::create_opaque_type<_S_NAME_>(#_S_NAME_);  \
		SP_DEFINE_STRUCT_DESC(_S_NAME_,__VA_ARGS__);        \
		return std::move(d_type);                                 \
	}                                                             \
};

}
// namespace simpla

#endif /* DATA_TYPE_H_ */
