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
#include <type_traits>

#include "check_concept.h"

namespace simpla
{

namespace traits
{

template<typename T> struct dimensions;
template<typename T> struct rank;
template<typename T, size_t> struct extent;

}  // namespace traits
template<typename, size_t...> struct nTuple;

template<typename _Tp, _Tp ... _I> struct integer_sequence;

template<typename T, size_t N>
struct nTuple<T, N>
{

	typedef T value_type;

	typedef T sub_type;

	typedef integer_sequence<size_t, N> dimensions;

	typedef value_type pod_type[N];

	static constexpr size_t dims = N;

	typedef nTuple<value_type, N> this_type;

	sub_type m_data_[dims];

	sub_type &operator[](size_t s)
	{
		return m_data_[s];
	}

	sub_type const &operator[](size_t s) const
	{
		return m_data_[s];
	}

	this_type &operator++()
	{
		++m_data_[N - 1];
		return *this;
	}

	this_type &operator--()
	{
		--m_data_[N - 1];
		return *this;
	}
	template<typename U, size_t ...I>
	operator nTuple<U,I...>() const
	{
		nTuple<U, I...> res;
		res = *this;
		return std::move(res);
	}

private:
	template<size_t I, typename TOp, typename ...Args>
	void foreach(std::integral_constant<size_t, I>, TOp const &op,
			Args && ...args)
	{
		op(std::forward<Args>(args)...);

		foreach(std::integral_constant<size_t, I - 1>(), op,
				std::forward<Args>(args)...);
	}
	template<size_t I, typename TOp, typename ...Args>
	void foreach(std::integral_constant<size_t, 0>, TOp const &op,
			Args && ...args)
	{
	}
public:

	template<typename TR>
	inline this_type &
	operator=(TR const &rhs)
	{

		//  assign different 'dimensions' ntuple
//		_seq_for<
//				min_not_zero<dims,
//						seq_get<0, typename nTuple_traits<TR>::dimensions>::value>::value
//
//		>::eval(_impl::_assign(), m_data_, rhs);

		foreach(std::integral_constant<size_t, dims>(), _impl::_assign(),
				m_data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type &
	operator=(TR const *rhs)
	{
		foreach(std::integral_constant<size_t, dims>(), _impl::_assign(),
				m_data_, rhs);
		return (*this);
	}

//	template<typename TR>
//	inline bool operator ==(TR const &rhs)
//	{
//		return _seq_reduce<
//				min_not_zero<dims,
//						seq_get<0, typename nTuple_traits<TR>::dimensions>::value>::value>::eval(
//				_impl::logical_and(), _impl::equal_to(), data_, rhs);;
//	}
//
	template<typename TR>
	inline this_type &operator+=(TR const &rhs)
	{
		foreach(std::integral_constant<size_t, dims>(), _impl::plus_assign(),
				m_data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type &operator-=(TR const &rhs)
	{
		foreach(std::integral_constant<size_t, dims>(), _impl::minus_assign(),
				m_data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type &operator*=(TR const &rhs)
	{
		foreach(std::integral_constant<size_t, dims>(),
				_impl::multiplies_assign(), m_data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type &operator/=(TR const &rhs)
	{
		foreach(std::integral_constant<size_t, dims>(), _impl::divides_assign(),
				m_data_, rhs);
		return (*this);
	}

//	template<size_t NR, typename TR>
//	void operator*(nTuple<NR, TR> const & rhs) = delete;
//
//	template<size_t NR, typename TR>
//	void operator/(nTuple<NR, TR> const & rhs) = delete;

};
template<typename _Tp, _Tp ... _I>
struct integer_sequence
{

	typedef integer_sequence<_Tp, _I...> type;

	typedef nTuple<_Tp, traits::extent<type, 0>::value> value_type;

	static constexpr value_type value = { _I... };

	static constexpr size_t size()
	{
		return traits::rank<type>::value;
	}
	constexpr operator value_type() const
	{
		return value;
	}

	constexpr value_type operator()() const
	{
		return value;
	}
};

template<typename _Tp, _Tp ... _I> constexpr typename
integer_sequence<_Tp, _I...>::value_type integer_sequence<_Tp, _I...>::value;

template<typename _Tp, _Tp _I>
struct integer_sequence<_Tp, _I> : public std::integral_constant<_Tp, _I>
{
};

/**
 * @ingroup utilities
 * @addtogroup type_traits Type traits
 * @{
 **/
template<typename _Tp, _Tp ... _I> struct integer_sequence;

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
template<typename T>
struct remove_all
{
	typedef typename std::remove_reference<typename std::remove_const<T>::type>::type type;
};

namespace traits
{

template<typename T>
struct rank: public std::integral_constant<size_t, std::rank<T>::value>
{
};
template<typename _Tp, _Tp ... _I>
struct rank<integer_sequence<_Tp, _I...>> : public std::integral_constant<
		size_t, 1>
{
};
template<typename _Tp, size_t N = 0>
struct extent: public std::extent<_Tp, N>
{
};
template<typename _Tp, _Tp ... _I>
struct extent<integer_sequence<_Tp, _I...>, 0> : public std::integral_constant<
		size_t, sizeof...(_I)>
{
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

template<typename T> struct value_type
{
	typedef T type;
};

}  // namespace traits
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

//template<typename T, typename TI, TI ...N>
//auto try_index(T & v, integer_sequence<TI, N...>)
//ENABLE_IF_DECL_RET_TYPE((!traits::is_indexable<T,TI>::value), (v))

//template<typename T, typename ...Args>
//auto try_index(std::shared_ptr<T> & v, Args &&... args)
//DECL_RET_TYPE( try_index(v.get(),std::forward<Args>(args)...))
//
//template<typename T, typename ...Args>
//auto try_index(std::shared_ptr<T> const & v, Args &&... args)
//DECL_RET_TYPE( try_index(v.get(),std::forward<Args>(args)...))

/// \note  http://stackoverflow.com/questions/3913503/metaprogram-for-bit-counting
template<unsigned long N> struct CountBits
{
	static const unsigned long n = CountBits<N / 2>::n + 1;
};

template<> struct CountBits<0>
{
	static const unsigned long n = 0;
};

inline unsigned long count_bits(unsigned long s)
{
	unsigned long n = 0;
	while (s != 0)
	{
		++n;
		s = s >> 1;
	}
	return n;
}

template<typename T> inline T* PointerTo(T & v)
{
	return &v;
}

template<typename T> inline T* PointerTo(T * v)
{
	return v;
}

template<typename TV, typename TR> inline TV TypeCast(TR const& obj)
{
	return std::move(static_cast<TV>(obj));
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

HAS_MEMBER_FUNCTION(begin)
HAS_MEMBER_FUNCTION(end)

template<typename T>
auto begin(T& l)
ENABLE_IF_DECL_RET_TYPE((has_member_function_begin<T>::value),( l.begin()))

template<typename T>
auto begin(T& l)
ENABLE_IF_DECL_RET_TYPE((!has_member_function_begin<T>::value),(std::get<0>(l)))

template<typename T>
auto begin(T const& l)
ENABLE_IF_DECL_RET_TYPE((has_member_function_begin<T>::value),( l.begin()))

template<typename T>
auto begin(T const& l)
ENABLE_IF_DECL_RET_TYPE((!has_member_function_begin<T>::value),(std::get<0>(l)))

template<typename T>
auto end(T& l)
ENABLE_IF_DECL_RET_TYPE((has_member_function_end<T>::value),( l.end()))

template<typename T>
auto end(T& l)
ENABLE_IF_DECL_RET_TYPE((!has_member_function_end<T>::value),(std::get<1>(l)))

template<typename T>
auto end(T const& l)
ENABLE_IF_DECL_RET_TYPE((has_member_function_end<T>::value),( l.end()))

template<typename T>
auto end(T const& l)
ENABLE_IF_DECL_RET_TYPE((!has_member_function_end<T>::value),(std::get<1>(l)))

HAS_MEMBER_FUNCTION(rbegin)
HAS_MEMBER_FUNCTION(rend)

template<typename T>
auto rbegin(T& l)
ENABLE_IF_DECL_RET_TYPE((has_member_function_begin<T>::value),( l.rbegin()))

template<typename T>
auto rbegin(T& l)
ENABLE_IF_DECL_RET_TYPE(
		(!has_member_function_begin<T>::value),(--std::get<1>(l)))

template<typename T>
auto rend(T& l)
ENABLE_IF_DECL_RET_TYPE((has_member_function_end<T>::value),( l.rend()))

template<typename T>
auto rend(T& l)
ENABLE_IF_DECL_RET_TYPE((!has_member_function_end<T>::value),(--std::get<0>(l)))

template<typename TI>
auto distance(TI const & b, TI const & e)
DECL_RET_TYPE((e-b))

template<typename _T>
struct is_iterator
{
private:
	typedef std::true_type yes;
	typedef std::false_type no;

	template<typename _U>
	static auto test(int) ->
	decltype(std::declval<_U>().operator *() );

	template<typename > static no test(...);

public:

	static constexpr bool value =
			!std::is_same<decltype(test<_T>(0)), no>::value;
};

template<typename TI>
auto ref(TI & it)
ENABLE_IF_DECL_RET_TYPE(is_iterator<TI>::value,(*it))
template<typename TI>
auto ref(TI & it)
ENABLE_IF_DECL_RET_TYPE(!is_iterator<TI>::value,(it))

HAS_MEMBER_FUNCTION(swap)

template<typename T> typename std::enable_if<has_member_function_swap<T>::value,
		void>::type sp_swap(T& l, T& r)
{
	l.swap(r);
}

template<typename T> typename std::enable_if<
		!has_member_function_swap<T>::value, void>::type sp_swap(T& l, T& r)
{
	std::swap(l, r);
}

template<typename > struct result_of;

template<typename F, typename ...Args> struct result_of<F(Args...)>
{
	typedef typename std::result_of<F(Args...)>::type type;
};

namespace _impl
{

struct GetValue
{
	template<typename TL, typename TI>
	constexpr auto operator()(TL const & v, TI const s) const
	DECL_RET_TYPE((try_index(v,s)))

	template<typename TL, typename TI>
	constexpr auto operator()(TL & v, TI const s) const
	DECL_RET_TYPE((try_index(v,s)))
};

} //namespace _impl
template<typename ...> struct index_of;

template<typename TC, typename TI>
struct index_of<TC, TI>
{
	typedef typename result_of<_impl::GetValue(TC, TI)>::type type;
};

HAS_MEMBER_FUNCTION(print)
template<typename TV>
auto sp_print(std::ostream & os,
		TV const & v)
		->typename std::enable_if<has_member_function_print<TV const,std::ostream &>::value,std::ostream &>::type
{
	return v.print(os);
}

template<typename TV>
auto sp_print(std::ostream & os,
		TV const & v)
		->typename std::enable_if<!has_member_function_print<TV const,std::ostream &>::value,std::ostream &>::type
{
	os << v;
	return os;
}

template<typename TI, TI L, TI R>
struct sp_max
{
	static constexpr TI value = L > R ? L : R;
};

template<typename TI, TI L, TI R>
struct sp_min
{
	static constexpr TI value = L < R ? L : R;
};
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

/**
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

/**
 * @} ingroup utilities
 */

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

//template<typename T, typename U>
//T & raw_cast(U& s)
//{
//	return *reinterpret_cast<T*>(&s);
//}
//template<typename T, typename U>
//T raw_cast(U&& s)
//{
//	return *reinterpret_cast<T*>(&s);
//}
//
//template<typename T, typename U>
//T assign_cast(U const & s)
//{
//	T res;
//	res = s;
//	return std::move(res);
//}

template<typename T>
T const & min(T const & first, T const & second)
{
	return std::min(first, second);
}

template<typename T>
T const & min(T const & first)
{
	return first;
}
template<typename T, typename ...Others>
T const & min(T const & first, Others &&... others)
{
	return min(first, min(std::forward<Others>(others)...));
}

template<typename T>
T const & max(T const & first, T const & second)
{
	return std::max(first, second);
}

template<typename T>
T const & max(T const & first)
{
	return first;
}

template<typename T, typename ...Others>
T const & max(T const & first, Others &&...others)
{
	return max(first, max(std::forward<Others>(others)...));
}

} // namespace simpla
#endif /* SP_TYPE_TRAITS_H_ */
