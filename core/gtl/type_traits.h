/**
 * @file type_traits.h
 *
 *  created on: 2014-6-15
 *      Author: salmon
 */

#ifndef SP_TYPE_TRAITS_H_
#define SP_TYPE_TRAITS_H_

#include <stddef.h>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

#include "check_concept.h"

namespace simpla
{
template<typename _Tp, _Tp... > struct integer_sequence;
template<typename, size_t...> struct nTuple;
/**
 * @ingroup utilities
 * @addtogroup type_traits Type traits
 * @{
 **/

typedef std::nullptr_t NullType;

struct EmptyType
{
};
struct do_nothing
{
	template<typename ...Args>
	void operator()(Args &&...) const
	{
	}
};

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

#define DECL_RET_TYPE(_EXPR_) ->decltype((_EXPR_)){return (_EXPR_);}

#define ENABLE_IF_DECL_RET_TYPE(_COND_,_EXPR_) \
        ->typename std::enable_if<_COND_,decltype((_EXPR_))>::type {return (_EXPR_);}

template<typename T>
struct rank
{
	static constexpr size_t value = std::rank<T>::value;
};

template<typename T> struct dimensions
{
	static constexpr size_t value[] = { 1 };
	typedef integer_sequence<size_t, 1> type;
};

template<typename T> constexpr size_t dimensions<T>::value[];

template<typename T> struct key_type
{
	typedef void type;
};

template<typename T> struct element_type
{
	typedef T type;
};

template<typename K, typename V, typename ...Others> struct key_type<
		std::map<K, V, Others...>>
{
	typedef K type;
};

template<typename K, typename V, typename ...Others> struct element_type<
		std::map<K, V, Others...>>
{
	typedef V type;
};

template<typename V, typename ...Others> struct key_type<
		std::vector<V, Others...>>
{
	typedef size_t type;
};

template<typename V, typename ...Others> struct element_type<
		std::vector<V, Others...>>
{
	typedef V type;
};

template<typename T, typename TI>
auto try_access(T & v,
		TI const& s)
				ENABLE_IF_DECL_RET_TYPE(
						! (check::is_indexable<T,TI>::value || check::is_shared_ptr<T >::value ) , (v))

template<typename T, typename TI>
auto try_access(T & v, TI const & s)
ENABLE_IF_DECL_RET_TYPE((check::is_indexable<T,TI>::value ), (v[s]))

template<typename T, typename TI>
auto try_access(T & v, TI const& s)
ENABLE_IF_DECL_RET_TYPE(( check::is_shared_ptr<T >::value ) , v.get()[s])

template<typename T, typename TI>
T & try_access(std::map<T, TI> & v, TI const& s)
{
	return v[s];
}
template<typename T, typename TI>
T const & try_access(std::map<T, TI> const& v, TI const& s)
{
	return v[s];
}
namespace _impl
{

template<size_t N>
struct recursive_try_access_aux
{
	template<typename T, typename TI>
	static auto eval(T & v, TI const *s)
	DECL_RET_TYPE(
			( recursive_try_access_aux<N-1>::eval(v[s[0]], s+1))
	)
};
template<>
struct recursive_try_access_aux<0>
{
	template<typename T, typename TI>
	static auto eval(T & v, TI const *s)
	DECL_RET_TYPE( ( v ) )
};
} // namespace _impl

template<typename T, typename TI>
auto try_access_r(T & v,
		TI const *s)
				ENABLE_IF_DECL_RET_TYPE((check::is_indexable<T,TI>::value),
						( _impl::recursive_try_access_aux<traits::rank<T>::value>::eval(v,s)))

template<typename T, typename TI>
auto try_access_r(T & v, TI const * s)
ENABLE_IF_DECL_RET_TYPE((!check::is_indexable<T,TI>::value), (v))

template<typename T, typename TI, size_t N>
auto try_access_r(T & v, nTuple<TI, N> const &s)
ENABLE_IF_DECL_RET_TYPE((check::is_indexable<T,TI>::value),
		( _impl::recursive_try_access_aux<N>::eval(v,s)))

template<typename T, typename TI, size_t N>
auto try_access_r(T & v, nTuple<TI, N> const &s)
ENABLE_IF_DECL_RET_TYPE((!check::is_indexable<T,TI>::value), (v))

template<typename T, typename TI, TI M, TI ...N>
auto try_access(T & v, integer_sequence<TI, M, N...>)
ENABLE_IF_DECL_RET_TYPE((check::is_indexable<T,TI>::value),
		try_access(v[M],integer_sequence<TI, N...>()))

template<typename T, typename TI, TI M, TI ...N>
auto try_access(T & v, integer_sequence<TI, M, N...>)
ENABLE_IF_DECL_RET_TYPE((!check::is_indexable<T,TI>::value), v)

//template<typename T, typename TI, TI ...N>
//auto try_access(T & v, integer_sequence<TI, N...>)
//ENABLE_IF_DECL_RET_TYPE((!check::is_indexable<T,TI>::value), (v))

//template<typename T, typename ...Args>
//auto try_access(std::shared_ptr<T> & v, Args &&... args)
//DECL_RET_TYPE( try_access(v.get(),std::forward<Args>(args)...))
//
//template<typename T, typename ...Args>
//auto try_access(std::shared_ptr<T> const & v, Args &&... args)
//DECL_RET_TYPE( try_access(v.get(),std::forward<Args>(args)...))

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
//}// namespace _impl
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

template<typename T>
struct sp_pod_traits
{
	typedef T type;

};
template<typename _Signature>
class sp_result_of
{
	typedef typename std::result_of<_Signature>::type _type;
public:
	typedef typename sp_pod_traits<_type>::type type;

};

template<unsigned int, typename ...> struct unpack_typelist;

template<typename T0, typename ...Others>
struct unpack_typelist<0, T0, Others...>
{
	typedef T0 type;
};
template<unsigned int N>
struct unpack_typelist<N>
{
	typedef void type;
};
template<unsigned int N, typename T0, typename ...Others>
struct unpack_typelist<N, T0, Others...>
{
	typedef typename unpack_typelist<N - 1, Others...>::type type;
};

template<typename, typename ...> struct find_type_in_list;

template<typename T>
struct find_type_in_list<T>
{
	static constexpr bool value = false;
};
template<typename T, typename U>
struct find_type_in_list<T, U>
{
	static constexpr bool value = std::is_same<T, U>::value;
};
template<typename T, typename U, typename ...Others>
struct find_type_in_list<T, U, Others...>
{
	static constexpr bool value = find_type_in_list<T, U>::value
			|| find_type_in_list<T, Others...>::value;
};
}  // namespace traits



template<typename TI>
auto ref(TI & it)
ENABLE_IF_DECL_RET_TYPE(check::is_iterator<TI>::value,(*it))
template<typename TI>
auto ref(TI & it)
ENABLE_IF_DECL_RET_TYPE(!check::is_iterator<TI>::value,(it))

template<typename > struct result_of;

template<typename F, typename ...Args> struct result_of<F(Args...)>
{
	typedef typename std::result_of<F(Args...)>::type type;
};

//namespace _impl
//{
//
//struct GetValue
//{
//	template<typename TL, typename TI>
//	constexpr auto operator()(TL const & v, TI const s) const
//	DECL_RET_TYPE((try_access(v,s)))
//
//	template<typename TL, typename TI>
//	constexpr auto operator()(TL & v, TI const s) const
//	DECL_RET_TYPE((try_access(v,s)))
//};
//
//} //namespace _impl
//template<typename ...> struct index_of;
//
//template<typename TC, typename TI>
//struct index_of<TC, TI>
//{
//	typedef typename result_of<_impl::GetValue(TC, TI)>::type type;
//};

/**
 * @} ingroup utilities
 */

}// namespace simpla
#endif /* SP_TYPE_TRAITS_H_ */
