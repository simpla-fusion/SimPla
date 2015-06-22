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

template<typename _Tp>
struct integer_sequence<_Tp>
{

public:
	typedef integer_sequence<_Tp> type;

	typedef _Tp value_type;

	static constexpr value_type value[] = { };

	static constexpr size_t size()
	{
		return 0;
	}

};
template<typename _Tp>
constexpr _Tp integer_sequence<_Tp>::value[];

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
template<typename T> struct type_id
{
	static std::string name()
	{
		return "unknown";
	}
};
template<typename T> struct reference
{
	typedef T type;
};
template<typename T> using reference_t=typename reference<T>::type;

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
};
template<typename T> using extents_t=typename extents<T>::type;

template<typename T, size_t N>
struct extents<T[N]> : public simpla::_impl::seq_concat<
		integer_sequence<size_t, N>, typename extents<T>::type>::type
{
};

template<typename T> struct key_type
{
	typedef size_t type;
};
template<typename T> using key_type_t=typename key_type<T>::type;

template<typename T> struct value_type
{
	typedef T type;
};
template<typename T> using value_type_t=typename value_type<T>::type;

template<typename T, typename TI>
auto index(T & v, TI const& s)
ENABLE_IF_DECL_RET_TYPE((!is_indexable<T,TI>::value ) , (v))

template<typename T, typename TI>
auto index(T & v, TI const & s)
ENABLE_IF_DECL_RET_TYPE((is_indexable<T,TI>::value ), (v[s]))

template<typename T, typename TI>
auto index(std::shared_ptr<T> & v, TI const& s)
DECL_RET_TYPE( v.get()[s])

template<typename T, typename TI>
auto index(std::shared_ptr<T> const& v, TI const& s)
DECL_RET_TYPE( v.get()[s])

template<typename T, typename TI>
auto index(T & v, integer_sequence<TI>)
ENABLE_IF_DECL_RET_TYPE((is_indexable<T,TI>::value),
		index(v ,integer_sequence<TI >()))

template<typename T, typename TI, TI M, TI ...N>
auto index(T & v, integer_sequence<TI, M, N...>)
ENABLE_IF_DECL_RET_TYPE((is_indexable<T,TI>::value),
		index(v[M],integer_sequence<TI, N...>()))

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
auto index(T & v,
		TI const *s)
				ENABLE_IF_DECL_RET_TYPE(( is_indexable<T,TI>::value),
						( _impl::recursive_try_index_aux<traits::rank<T>::value>::eval(v,s)))

template<typename T, typename TI, size_t N>
auto index(T & v, nTuple<TI, N> const &s)
ENABLE_IF_DECL_RET_TYPE(( is_indexable<T,TI>::value),
		( _impl::recursive_try_index_aux<N>::eval(v,s)))

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

template<typename T>
struct pod_type
{
	typedef T type;
};
template<typename T> using pod_type_t = typename pod_type<T>::type;

template<typename T>
struct primary_type
{
	typedef T type;
};
template<typename T> using primary_type_t=typename primary_type<T>::type;

template<typename _Signature>
struct result_of
{
	typedef typename std::result_of<_Signature>::type type;

};
template<typename _Signature> using result_of_t = typename result_of<_Signature>::type;

template<typename T0>
T0 const& max(T0 const& first)
{
	return first;
}
template<typename T0, typename ...Others>
T0 const& max(T0 const& first, Others const& ...others)
{
	return std::max(first, max(others...));
}

template<typename T0>
T0 const& min(T0 const& first)
{
	return first;
}
template<typename T0, typename ...Others>
T0 const& min(T0 const& first, Others const& ...others)
{
	return std::min(first, min(others...));
}
template<size_t N, typename T> struct access;

template<size_t N, typename ...T>
struct access<N, std::tuple<T...>>
{
	static constexpr auto get(std::tuple<T...>& v)
	DECL_RET_TYPE(std::get<N>(v))

	static constexpr auto get(std::tuple<T...> const& v)
	DECL_RET_TYPE(std::get<N>(v))

	template<typename U>
	static void set(std::tuple<T...>& v, U const &u)
	{
		get(v) = u;
	}
};

template<size_t N, typename T>
struct access<N, T*>
{
	static constexpr auto get(T* v)
	DECL_RET_TYPE(v[N])

	static constexpr auto get(T const* v)
	DECL_RET_TYPE(v[N])

	template<typename U>
	static void set(T* v, U const &u)
	{
		get(v) = u;
	}
};

template<size_t N, typename T0, typename T1>
struct access<N, std::pair<T0, T1>>
{
	static constexpr auto get(std::pair<T0, T1>& v)
	DECL_RET_TYPE(std::get<N>(v))

	static constexpr auto get(std::pair<T0, T1> const& v)
	DECL_RET_TYPE(std::get<N>(v))

	template<typename U>
	static void set(std::pair<T0, T1>& v, U const &u)
	{
		get(v) = u;
	}
};
namespace _impl
{

template<size_t ...N> struct access_helper;

template<size_t N0, size_t ...N>
struct access_helper<N0, N...>
{

	template<typename T>
	static constexpr auto get(T const& v)
	DECL_RET_TYPE((access_helper< N... >::get(
							access_helper<N0>::get( (v)))))

	template<typename T>
	static constexpr auto get(T & v)
	DECL_RET_TYPE((access_helper< N... >::get(
							access_helper<N0>::get( (v)))))
	template<typename T, typename U>
	static void set(T & v, U const &u)
	{
		access_helper<N0, N...>::get(v) = u;
	}

};

template<size_t N>
struct access_helper<N>
{
	template<typename T>
	static constexpr auto get(T& v)
	DECL_RET_TYPE((traits::access<N, T>::get(v)))

	template<typename T>
	static constexpr auto get(T const& v)
	DECL_RET_TYPE((traits::access<N, T>::get(v)))

	template<typename T, typename U>
	static void set(T& v, U const &u)
	{
		return traits::access<N, T>::get(v, u);
	}

};

template<>
struct access_helper<>
{
	template<typename T>
	static constexpr T& get(T& v)
	{
		return v;
	}
	template<typename T>
	static constexpr T const& get(T const& v)
	{
		return v;
	}
	template<typename T, typename U>
	static void set(T& v, U const &u)
	{
		v = u;
	}

};
}  // namespace _impl
template<size_t N, typename ...T>
auto get(std::tuple<T...> & v)
DECL_RET_TYPE(std::get<N>(v))

template<size_t ...N, typename T>
auto get(T & v)
DECL_RET_TYPE(( _impl::access_helper< N...>::get(v)))
template<size_t ...N, typename T>
auto get(T const& v)
DECL_RET_TYPE(( _impl::access_helper< N...>::get(v)))
}
// namespace traits

//template<typename T, typename TI>
//auto try_index(T & v, TI const& s)
//ENABLE_IF_DECL_RET_TYPE((!traits::is_indexable<T,TI>::value ) , (v))
//
//template<typename T, typename TI>
//auto try_index(T & v, TI const & s)
//ENABLE_IF_DECL_RET_TYPE((traits::is_indexable<T,TI>::value ), (v[s]))
//
//template<typename T, typename TI>
//auto try_index(std::shared_ptr<T> & v, TI const& s)
//DECL_RET_TYPE( v.get()[s])
//
//template<typename T, typename TI>
//auto try_index(std::shared_ptr<T> const& v, TI const& s)
//DECL_RET_TYPE( v.get()[s])
//
//template<typename T, typename TI>
//T & try_index(std::map<T, TI> & v, TI const& s)
//{
//	return v[s];
//}
//template<typename T, typename TI>
//T const & try_index(std::map<T, TI> const& v, TI const& s)
//{
//	return v[s];
//}
//
//template<typename T, typename TI, TI M, TI ...N>
//auto try_index(T & v, integer_sequence<TI, M, N...>)
//ENABLE_IF_DECL_RET_TYPE((traits::is_indexable<T,TI>::value),
//		try_index(v[M],integer_sequence<TI, N...>()))
//
//template<typename T, typename TI, TI M, TI ...N>
//auto try_index(T & v, integer_sequence<TI, M, N...>)
//ENABLE_IF_DECL_RET_TYPE((!traits::is_indexable<T,TI>::value), v)

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

}// namespace simpla
#endif /* SP_TYPE_TRAITS_H_ */
