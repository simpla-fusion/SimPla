/**
 * @file type_traits_ext.h
 *
 *  Created on: 2015年6月11日
 *      Author: salmon
 */

#ifndef CORE_GTL_TYPE_TRAITS_EXT_H_
#define CORE_GTL_TYPE_TRAITS_EXT_H_

namespace simpla
{

namespace traits
{

template<typename T>
struct remove_all
{
	typedef typename std::remove_reference<typename std::remove_const<T>::type>::type type;
};

/**
 * @name Replace Type
 * @{
 */

template<size_t, typename ...> struct replace_template_type;

template<typename TV, typename T0, typename ...Others, template<typename ...> class TT>
struct replace_template_type<0,TV,TT<T0, Others...> >
{
	typedef TT< TV,Others...> type;
};

template<typename TV, template<typename ...> class TT, typename T0,typename T1,typename ...Others>
struct replace_template_type<1,TV,TT<T0,T1,Others...> >
{
	typedef TT<T0,TV,Others...> type;
};
/**
 * @}
 */


template<typename T> inline T* PointerTo(T & v)
{
	return &v;
}

template<typename T> inline T* PointerTo(T * v)
{
	return v;
}

template<int...> class int_tuple_t;

//namespace _impl
//{
////******************************************************************************************************
//// Third-part code begin
//// ref: https://gitorious.org/redistd/redistd
//// Copyright Jonathan Wakely 2012
//// Distributed under the Boost Software License, Version 1.0.
//// (See accompanying file LICENSE_1_0.txt or copy at
//// http://www.boost.org/LICENSE_1_0.txt)
//
///// A type that represents a parameter pack of zero or more integers.
//template<unsigned ... Indices>
//struct index_tuple
//{
//	/// Generate an index_tuple with an additional element.
//	template<unsigned N>
//	using append = index_tuple<Indices..., N>;
//};
//
///// Unary metafunction that generates an index_tuple containing [0, Size)
//template<unsigned Size>
//struct make_index_tuple
//{
//	typedef typename make_index_tuple<Size - 1>::type::template append<Size - 1> type;
//};
//
//// Terminal case of the recursive metafunction.
//template<>
//struct make_index_tuple<0u>
//{
//	typedef index_tuple<> type;
//};
//
//template<typename ... Types>
//using to_index_tuple = typename make_index_tuple<sizeof...(Types)>::type;
//// Third-part code end
////******************************************************************************************************
//

//
//HAS_MEMBER_FUNCTION(begin)
//HAS_MEMBER_FUNCTION(end)
//
//template<typename T>
//auto begin(T& l)
//ENABLE_IF_DECL_RET_TYPE((has_member_function_begin<T>::value),( l.begin()))
//
//template<typename T>
//auto begin(T& l)
//ENABLE_IF_DECL_RET_TYPE((!has_member_function_begin<T>::value),(std::get<0>(l)))
//
//template<typename T>
//auto begin(T const& l)
//ENABLE_IF_DECL_RET_TYPE((has_member_function_begin<T>::value),( l.begin()))
//
//template<typename T>
//auto begin(T const& l)
//ENABLE_IF_DECL_RET_TYPE((!has_member_function_begin<T>::value),(std::get<0>(l)))
//
//template<typename T>
//auto end(T& l)
//ENABLE_IF_DECL_RET_TYPE((has_member_function_end<T>::value),( l.end()))
//
//template<typename T>
//auto end(T& l)
//ENABLE_IF_DECL_RET_TYPE((!has_member_function_end<T>::value),(std::get<1>(l)))
//
//template<typename T>
//auto end(T const& l)
//ENABLE_IF_DECL_RET_TYPE((has_member_function_end<T>::value),( l.end()))
//
//template<typename T>
//auto end(T const& l)
//ENABLE_IF_DECL_RET_TYPE((!has_member_function_end<T>::value),(std::get<1>(l)))
//
//HAS_MEMBER_FUNCTION(rbegin)
//HAS_MEMBER_FUNCTION(rend)
//
//template<typename T>
//auto rbegin(T& l)
//ENABLE_IF_DECL_RET_TYPE((has_member_function_begin<T>::value),( l.rbegin()))
//
//template<typename T>
//auto rbegin(T& l)
//ENABLE_IF_DECL_RET_TYPE(
//		(!has_member_function_begin<T>::value),(--std::get<1>(l)))
//
//template<typename T>
//auto rend(T& l)
//ENABLE_IF_DECL_RET_TYPE((has_member_function_end<T>::value),( l.rend()))
//
//template<typename T>
//auto rend(T& l)
//ENABLE_IF_DECL_RET_TYPE((!has_member_function_end<T>::value),(--std::get<0>(l)))
//
//template<typename TI>
//auto distance(TI const & b, TI const & e)
//DECL_RET_TYPE((e-b))

//}// namespace _impl


template<typename TI>
auto ref(TI & it)
ENABLE_IF_DECL_RET_TYPE(check::is_iterator<TI>::value,(*it))
template<typename TI>
auto ref(TI & it)
ENABLE_IF_DECL_RET_TYPE(!check::is_iterator<TI>::value,(it))
}  // namespace traits

}  // namespace simpla

#endif /* CORE_GTL_TYPE_TRAITS_EXT_H_ */
