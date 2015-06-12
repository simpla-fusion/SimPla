/**
 * @file type_traits.h
 *
 *  created on: 2014-6-15
 *      Author: salmon
 */

#ifndef SP_TYPE_TRAITS_H_
#define SP_TYPE_TRAITS_H_

#include <stddef.h>
#include <map>
#include <tuple>
#include <type_traits>

#include "check_concept.h"
#include "macro.h"

namespace simpla
{
//typedef std::nullptr_t NullType;
//
//struct EmptyType
//{
//};
//struct do_nothing
//{
//	template<typename ...Args>
//	void operator()(Args &&...) const
//	{
//	}
//};
//template<typename T>
//struct remove_all
//{
//	typedef typename std::remove_reference<typename std::remove_const<T>::type>::type type;
//};
template<typename _Tp, _Tp ... _I> struct integer_sequence;
template<typename, size_t...>struct nTuple;
//////////////////////////////////////////////////////////////////////
/// integer_sequence
//////////////////////////////////////////////////////////////////////

/**
 *  alt. of std::integer_sequence ( C++14)
 *  @quto http://en.cppreference.com/w/cpp/utility/integer_sequence
 *  The class template  integer_sequence represents a
 *  compile-time sequence of integers. When used as an argument
 *   to a function template, the parameter pack Ints can be deduced
 *   and used in pack expansion.
 */
template<typename _Tp, _Tp ... _I>
struct integer_sequence
{
private:
	static constexpr size_t m_size_ = sizeof...(_I);

public:
	typedef integer_sequence<_Tp, _I...> type;

	typedef _Tp value_type;

	static constexpr value_type value[] = { _I... };

	static constexpr size_t size()
	{
		return m_size_;
	}

};

template<typename _Tp, _Tp ... _I> constexpr typename
integer_sequence<_Tp, _I...>::value_type integer_sequence<_Tp, _I...>::value[];

template<size_t ... Ints>
using index_sequence = integer_sequence<std::size_t, Ints...>;

namespace _impl
{

/**
 *  cat two tuple/integer_sequence
 */
template<typename ...> struct seq_concat;

template<typename First, typename ...Others>
struct seq_concat<First, Others...>
{
	typedef typename seq_concat<First, typename seq_concat<Others...>::type>::type type;
};
template<typename First>
struct seq_concat<First>
{
	typedef First type;
};

template<typename _Tp, _Tp ... _M, _Tp ... _N>
struct seq_concat<integer_sequence<_Tp, _M...>, integer_sequence<_Tp, _N...> >
{
	typedef integer_sequence<_Tp, _M..., _N...> type;
};

template<typename Func, typename Tup, size_t ... index>
auto invoke_helper(Func&& func, Tup&& tup, index_sequence<index...>)
DECL_RET_TYPE(func(std::get<index>(std::forward<Tup>(tup))...))

template<typename TP, size_t N>
struct gen_seq
{
	typedef typename seq_concat<typename gen_seq<TP, N - 1>::type,
			integer_sequence<TP, N - 1> >::type type;
};

template<typename TP>
struct gen_seq<TP, 0UL>
{
	typedef integer_sequence<TP> type;
};
} // namespace _impl

template<class T, T N>
using make_integer_sequence =typename _impl::gen_seq<T,N>::type;

template<size_t N>
using make_index_sequence = make_integer_sequence< size_t, N>;

template<typename Func, typename Tup>
auto invoke(Func&& func,
		Tup&& tup)
				DECL_RET_TYPE(

						_impl::invoke_helper(std::forward<Func>(func),
								std::forward<Tup>(tup),
								make_index_sequence<std::tuple_size<typename std::decay<Tup>::type>::value>()
						)
				)

////////////////////////////////////////////////////////////////////////
///// Property queries of n-dimensional array
////////////////////////////////////////////////////////////////////////
//
//template<typename, size_t...> struct nTuple;
//
namespace traits
{
template<typename T> struct reference
{
	typedef T type;
};

/**
 *  alt. of std::rank
 *  @quto http://en.cppreference.com/w/cpp/types/rank
 *  If T is an array type, provides the member constant
 *  value equal to the number of dimensions of the array.
 *  For any other type, value is 0.
 */
template<typename T>
struct rank: public std::integral_constant<size_t, std::rank<T>::value>
{

};

/**
 * alt. of std::extent
 *  @quto http://en.cppreference.com/w/cpp/types/extent
 *  If T is an array type, provides the member constant value equal to
 * the number of elements along the Nth dimension of the array, if N
 * is in [0, std::rank<T>::value). For any other type, or if T is array
 * of unknown bound along its first dimension and N is 0, value is 0.
 */
template<typename T, size_t N = 0>
struct extent: public std::integral_constant<size_t, std::extent<T, N>::value>
{
};

/**
 * integer sequence of the number of element along all dimensions
 * i.e.
 *
 */
template<typename T>
struct extents: public integer_sequence<size_t>
{
	typedef integer_sequence<size_t> type;

};

template<int N, typename T0>
auto get(T0 & v)
DECL_RET_TYPE (std::get<N>(v))

template<typename T> struct key_type
{
	typedef size_t type;
};

template<typename T> struct value_type
{
	typedef T type;
};

template<typename T, size_t N>
struct extents<T[N]> : public simpla::_impl::seq_concat<
		integer_sequence<size_t, N>, typename extents<T>::type>::type
{
	typedef typename simpla::_impl::seq_concat<integer_sequence<size_t, N>,
			typename extents<T>::type>::type type;
};

} // namespace traits
//
///**
// * @name Replace Type
// * @{
// */
//
//template<size_t, typename ...> struct replace_template_type;
//
//template<typename TV, typename T0, typename ...Others, template<typename ...> class TT>
//struct replace_template_type<0,TV,TT<T0, Others...> >
//{
//	typedef TT< TV,Others...> type;
//};
//
//template<typename TV, template<typename ...> class TT, typename T0,typename T1,typename ...Others>
//struct replace_template_type<1,TV,TT<T0,T1,Others...> >
//{
//	typedef TT<T0,TV,Others...> type;
//};
///**
// * @}
// */
//
template<typename T, typename TI>
auto try_index(T & v,
		TI const& s)
				ENABLE_IF_DECL_RET_TYPE(
						! (traits::is_indexable<T,TI>::value || traits::is_shared_ptr<T >::value ) , (v))

template<typename T, typename TI>
auto try_index(T & v, TI const & s)
ENABLE_IF_DECL_RET_TYPE((traits::is_indexable<T,TI>::value ), (v[s]))

template<typename T, typename TI>
auto try_index(T & v, TI const& s)
ENABLE_IF_DECL_RET_TYPE(( traits::is_shared_ptr<T >::value ) , v.get()[s])

template<typename T, typename TI>
T & try_index(std::map<T, TI> & v, TI const& s)
{
	return v[s];
}
template<typename T, typename TI>
T const & try_index(std::map<T, TI> const& v, TI const& s)
{
	return v[s];
}
namespace _impl
{

template<size_t N>
struct recursive_try_index_aux
{
	template<typename T, typename TI>
	static auto eval(T & v, TI const *s)
	DECL_RET_TYPE(
			( recursive_try_index_aux<N-1>::eval(v[s[0]], s+1))
	)
};
template<>
struct recursive_try_index_aux<0>
{
	template<typename T, typename TI>
	static auto eval(T & v, TI const *s)
	DECL_RET_TYPE( ( v ) )
};
} // namespace _impl

template<typename T, typename TI>
auto try_index_r(T & v,
		TI const *s)
				ENABLE_IF_DECL_RET_TYPE((traits::is_indexable<T,TI>::value),
						( _impl::recursive_try_index_aux<traits::rank<T>::value>::eval(v,s)))

template<typename T, typename TI>
auto try_index_r(T & v, TI const * s)
ENABLE_IF_DECL_RET_TYPE((!traits::is_indexable<T,TI>::value), (v))

template<typename T, typename TI, size_t N>
auto try_index_r(T & v, nTuple<TI, N> const &s)
ENABLE_IF_DECL_RET_TYPE((traits::is_indexable<T,TI>::value),
		( _impl::recursive_try_index_aux<N>::eval(v,s)))

template<typename T, typename TI, size_t N>
auto try_index_r(T & v, nTuple<TI, N> const &s)
ENABLE_IF_DECL_RET_TYPE((!traits::is_indexable<T,TI>::value), (v))

template<typename T, typename TI, TI M, TI ...N>
auto try_index(T & v, integer_sequence<TI, M, N...>)
ENABLE_IF_DECL_RET_TYPE((traits::is_indexable<T,TI>::value),
		try_index(v[M],integer_sequence<TI, N...>()))

template<typename T, typename TI, TI M, TI ...N>
auto try_index(T & v, integer_sequence<TI, M, N...>)
ENABLE_IF_DECL_RET_TYPE((!traits::is_indexable<T,TI>::value), v)

////template<typename T, typename TI, TI ...N>
////auto try_index(T & v, integer_sequence<TI, N...>)
////ENABLE_IF_DECL_RET_TYPE((!traits::is_indexable<T,TI>::value), (v))
//
////template<typename T, typename ...Args>
////auto try_index(std::shared_ptr<T> & v, Args &&... args)
////DECL_RET_TYPE( try_index(v.get(),std::forward<Args>(args)...))
////
////template<typename T, typename ...Args>
////auto try_index(std::shared_ptr<T> const & v, Args &&... args)
////DECL_RET_TYPE( try_index(v.get(),std::forward<Args>(args)...))
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
//
//template<typename _T>
//struct is_iterator
//{
//private:
//	typedef std::true_type yes;
//	typedef std::false_type no;
//
//	template<typename _U>
//	static auto test(int) ->
//	decltype(std::declval<_U>().operator *() );
//
//	template<typename > static no test(...);
//
//public:
//
//	static constexpr bool value =
//			!std::is_same<decltype(test<_T>(0)), no>::value;
//};
//
///**
// * @} ingroup utilities
// */
//
namespace traits
{
namespace _impl
{

template<size_t N>
struct unpack_args_helper
{
	template<typename ... Args>
	auto eval(Args && ...args)
	DECL_RET_TYPE(unpack_args_helper<N-1>(std::forward<Args>(args)...))
};
template<>
struct unpack_args_helper<0>
{
	template<typename First, typename ... Args>
	auto eval(First && first, Args && ...args)
	DECL_RET_TYPE( std::forward<First>(first) )

};
}  // namespace _impl

template<size_t N, typename ... Args>
auto unpack_args(Args && ...args)
DECL_RET_TYPE ((_impl::unpack_args_helper<N>(std::forward<Args> (args)...)))

template<size_t N, typename _TP, _TP ...I> struct unpack_int_seq;

template<typename _Tp, _Tp I0, _Tp ...I>
struct unpack_int_seq<0, _Tp, I0, I...> : public std::integral_constant<_Tp, I0>
{

};
template<size_t N, typename _Tp, _Tp I0, _Tp ...I>
struct unpack_int_seq<N, _Tp, I0, I...> : public std::integral_constant<_Tp,
		unpack_int_seq<N - 1, _Tp, I...>::value>
{
};

template<size_t N, typename _Tp>
struct unpack_int_seq<N, _Tp> : public std::integral_constant<_Tp, 0>
{
};

template<unsigned int, typename ...> struct unpack_type_seq;

template<typename T0, typename ...Others>
struct unpack_type_seq<0, T0, Others...>
{
	typedef T0 type;
};
template<unsigned int N>
struct unpack_type_seq<N>
{
	typedef void type;
};
template<unsigned int N, typename T0, typename ...Others>
struct unpack_type_seq<N, T0, Others...>
{
	typedef typename unpack_type_seq<N - 1, Others...>::type type;
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

} // namespace simpla
#endif /* SP_TYPE_TRAITS_H_ */
