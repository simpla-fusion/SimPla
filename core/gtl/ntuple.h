/**
 * Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id: nTuple.h 990 2010-12-14 11:06:21Z salmon $
 * @file ntuple.h
 *
 *  created on: Jan 27, 2010
 *      Author: yuzhi
 */

#ifndef CORE_GTL_NTUPLE_H_
#define CORE_GTL_NTUPLE_H_

#include <cstdbool>
#include <cstddef>
#include <type_traits>

#include "expression_template.h"
#include "macro.h"
#include "mpl.h"
#include "type_traits.h"

namespace simpla
{

/**
 * @ingroup gtl
 * @addtogroup ntuple n-tuple
 * @{
 *
 * @brief nTuple :n-tuple
 *
 * Semantics:
 *    n-tuple is a sequence (or ordered list) of n elements, where n is a positive
 *    integral. There is also one 0-tuple, an empty sequence. An n-tuple is defined
 *    inductively using the construction of an ordered pair. Tuples are usually
 *    written by listing the elements within parenthesis '( )' and separated by
 *    commas; for example, (2, 7, 4, 1, 7) denotes a 5-tuple.
 *      [ wiki http://en.wikipedia.org/wiki/Tuple]
 *
 * Implement:
 *
 * @code{
 *   template<typename T, size_t ... n> struct nTuple;
 *
 *   nTuple<double,5> t={1,2,3,4,5};
 *
 *   nTuple<T,N...> primary ntuple
 *
 *   nTuple<Expression<TOP,TExpr>> unary nTuple expression
 *
 *   nTuple<Expression<TOP,TExpr1,TExpr2>> binary nTuple expression
 *
 *
 *
 *   nTuple<T,N> equiv. build-in array T[N]
 *
 *   nTuple<T,N,M> equiv. build-in array T[N][M]
 *
 *    @endcode}
 **/

/// n-dimensional primary type
template<typename, size_t...> struct nTuple;

template<typename ...> class Expression;

namespace traits
{
template<typename> struct primary_type;
template<typename> struct pod_type;
template<typename>
struct is_ntuple
{
	static constexpr bool value = false;
};

template<typename T, size_t ...N>
struct is_ntuple<nTuple<T, N...>>
{
	static constexpr bool value = true;
};

template<typename T, size_t M, size_t ...N>
struct reference<nTuple<T, M, N...>>
{
	typedef nTuple<T, M, N...> const &type;
};
template<typename T>
struct reference<nTuple<T>>
{
	typedef nTuple<T> type;
};


template<typename  ...T, typename TI>
auto index(nTuple<Expression<T...>> const &v, TI const &s)
DECL_RET_TYPE((v[s]))

}  // namespace traits




template<typename TV>
struct nTuple<TV>
{
	typedef TV value_type;

	typedef void sub_type;

	typedef integer_sequence<size_t> extents;

	typedef value_type pod_type;

};


template<typename ... T>
struct nTuple<Expression<T...>> : public Expression<T...>
{
	typedef nTuple<Expression<T...>> this_type;

	using Expression<T...>::m_op_;
	using Expression<T...>::args;
	using Expression<T...>::Expression;

	template<typename U, size_t ...N>
	operator nTuple<U, N...>() const
	{
		nTuple<U, N...> res;
		res = *this;
		return std::move(res);
	}


	template<typename ID>
	inline auto operator[](ID const &s) const
	DECL_RET_TYPE (at(s))

private:
	template<typename ID, size_t ... index>
	auto _invoke_helper(ID s, index_sequence<index...>) const
	DECL_RET_TYPE(m_op_(traits::index(std::get<index>(args), s)...))

public:
	template<typename ID>
	auto at(ID const &s) const
	DECL_RET_TYPE((
			_invoke_helper(s,
					typename make_index_sequence<sizeof...(T) - 1>::type())))


	template<typename ID>
	inline auto operator[](ID const &s) const
	DECL_RET_TYPE((
			_invoke_helper(s,
					typename make_index_sequence<sizeof...(T) - 1>::type())))


};


template<typename TOP, typename ARG0>
struct nTuple<Expression<TOP, ARG0>> : public Expression<TOP, ARG0>
{
	typedef nTuple<Expression<TOP, ARG0>> this_type;

	using Expression<TOP, ARG0>::m_op_;
	using Expression<TOP, ARG0>::args;
	using Expression<TOP, ARG0>::Expression;

	template<typename U, size_t ...N>
	operator nTuple<U, N...>() const
	{
		nTuple<U, N...> res;
		res = *this;
		return std::move(res);
	}


	template<typename ID>
	auto at(ID const &s) const
	DECL_RET_TYPE(m_op_(traits::index(std::get<0>(args), s)))

	template<typename ID>
	inline auto operator[](ID const &s) const
	DECL_RET_TYPE(m_op_(traits::index(std::get<0>(args), s)))


};

template<typename TOP, typename ARG0, typename ARG1>
struct nTuple<Expression<TOP, ARG0, ARG1>> : public Expression<TOP, ARG0, ARG1>
{
	typedef nTuple<Expression<TOP, ARG0, ARG1>> this_type;

	using Expression<TOP, ARG0, ARG1>::m_op_;
	using Expression<TOP, ARG0, ARG1>::args;
	using Expression<TOP, ARG0, ARG1>::Expression;

	template<typename U, size_t ...N>
	operator nTuple<U, N...>() const
	{
		nTuple<U, N...> res;
		res = *this;
		return std::move(res);
	}


	template<typename ID>
	auto at(ID const &s) const
	DECL_RET_TYPE(m_op_(traits::index(std::get<0>(args), s), traits::index(std::get<1>(args), s)))

	template<typename ID>
	inline auto operator[](ID const &s) const
	DECL_RET_TYPE(m_op_(traits::index(std::get<0>(args), s), traits::index(std::get<1>(args), s)))


};


template<typename TV, size_t N, size_t ...M>
struct nTuple<TV, N, M...>
{
private:

	typedef TV value_type;

	typedef nTuple<value_type, N, M...> this_type;

	static constexpr size_t m_extent = N;

public:

	typedef typename std::conditional<(sizeof...(M) == 0), value_type,
			nTuple<value_type, M...>>::type sub_type;

	sub_type data_[m_extent];

	sub_type &operator[](size_t s)
	{
		return data_[s];
	}

	sub_type const &operator[](size_t s) const
	{
		return data_[s];
	}


	sub_type &at(size_t s)
	{
		return data_[s];
	}

	sub_type const &at(size_t s) const
	{
		return data_[s];
	}

	this_type &operator++()
	{
		++data_[N - 1];
		return *this;
	}

	this_type &operator--()
	{
		--data_[N - 1];
		return *this;
	}

	template<typename U, size_t ...I>
	operator nTuple<U, I...>() const
	{
		nTuple<U, I...> res;
		res = *this;
		return std::move(res);
	}

public:

	template<typename TR>
	inline this_type &
	operator=(TR const &rhs)
	{
		assign(_impl::_assign(), rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type &
	operator=(TR const *rhs)
	{
		assign(_impl::_assign(), rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type &operator+=(TR const &rhs)
	{
		assign(_impl::plus_assign(), rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type &operator-=(TR const &rhs)
	{
		assign(_impl::minus_assign(), rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type &operator*=(TR const &rhs)
	{
		assign(_impl::multiplies_assign(), rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type &operator/=(TR const &rhs)
	{

		assign(_impl::divides_assign(), rhs);
		return (*this);
	}

private:

	template<typename Op, typename TR>
	void assign(Op const &op, TR const &rhs)
	{
		for (int s = 0; s < N; ++s)
		{
			op(data_[s], traits::index(rhs, s));
		}

	}

};

namespace traits
{


template<typename T, size_t ...M, size_t N>
struct access<N, nTuple<T, M...> >
{

	static constexpr auto get(nTuple<T, M...> &v)
	DECL_RET_TYPE(v[N])

	static constexpr auto get(nTuple<T, M...> const &v)
	DECL_RET_TYPE(v[N])

	template<typename U>
	static void set(nTuple<T, M...> &v, U const &u)
	{
		get(v) = u;
	}

};
//template<size_t M, typename T, size_t ...N>
//auto get(nTuple<T, N...> const& v)
//DECL_RET_TYPE((v[M]))
//template<size_t M, typename T, size_t ...N>
//auto get(nTuple<T, N...> & v)
//DECL_RET_TYPE((v[M]))
/**
 * C++11 <type_traits>
 * @ref http://en.cppreference.com/w/cpp/types/rank
 */

template<typename T, size_t ...N>
struct rank<nTuple<T, N...>> : public std::integral_constant<size_t,
		extents_t<nTuple<T, N...>>::size()>
{
};
template<typename TV, size_t ...M, size_t N>
struct extent<nTuple<TV, M...>, N> : public mpl::unpack_int_seq<N, size_t, M...>::type
{
};
template<typename T, size_t ...N>
struct key_type<nTuple<T, N...>>
{
	typedef size_t type;
};

namespace _impl
{

template<typename ...> struct make_pod_array;
template<typename ...> struct make_primary_nTuple;

template<typename TV, typename TI>
struct make_pod_array<TV, integer_sequence<TI>>
{
	typedef TV type;
};
template<typename TV, typename TI, TI N0, TI ... N>
struct make_pod_array<TV, integer_sequence<TI, N0, N...>>
{
	typedef typename make_pod_array<TV, integer_sequence<TI, N...>>::type type[N0];
};

template<typename TV, typename TI, TI ... N>
struct make_primary_nTuple<TV, integer_sequence<TI, N...>>
{
	typedef nTuple<TV, N...> type;
};
template<typename TV, typename TI>
struct make_primary_nTuple<TV, integer_sequence<TI>>
{
	typedef TV type;
};

template<typename ... T> using make_pod_array_t = typename make_pod_array<T...>::type;
template<typename ... T> using make_primary_nTuple_t = typename make_primary_nTuple<T...>::type;
}
// namespace _impl

template<typename T, size_t ...N>
struct primary_type<nTuple<T, N...>>
{
	typedef _impl::make_primary_nTuple_t<

			traits::value_type_t<nTuple<T, N...>>,

			traits::extents_t<nTuple<T, N...>>
	>
			type;

};

template<typename T> using ntuple_cast_t=typename primary_type<T>::type;

template<typename T, size_t ...N>
struct pod_type<nTuple<T, N...>>
{
	typedef _impl::make_pod_array_t<

			traits::value_type_t<nTuple<T, N...>>,

			traits::extents_t<nTuple<T, N...>>> type;

};


template<typename TV, size_t ...M>
struct extents<nTuple<TV, M...> > : public simpla::traits::seq_concat<
		integer_sequence<size_t, M...>, traits::extents_t<TV>>
{
};


namespace _impl
{
template<typename ...> struct extents_helper;

template<typename TOP>
struct extents_helper<TOP>
{
	typedef integer_sequence<size_t> type;
};

template<typename TOP, typename First, typename ...Others>
struct extents_helper<TOP, First, Others...>
{
	typedef typename extents_helper<TOP, First,
			typename extents_helper<TOP, Others...>::type>::type type;
};
template<typename TOP, typename _Tp, _Tp ...N>
struct extents_helper<TOP, integer_sequence<_Tp, N...>, integer_sequence<_Tp> >
{
	typedef integer_sequence<_Tp, N...> type;
};

template<typename TOP, typename _Tp, _Tp ...N>
struct extents_helper<TOP, integer_sequence<_Tp>, integer_sequence<_Tp, N...> >
{
	typedef integer_sequence<_Tp, N...> type;
};

template<typename TOP, typename _Tp>
struct extents_helper<TOP, integer_sequence<_Tp>, integer_sequence<_Tp> >
{
	typedef integer_sequence<_Tp> type;
};
template<typename TOP, typename _Tp, _Tp ...N, _Tp ...M>
struct extents_helper<TOP, integer_sequence<_Tp, N...>,
		integer_sequence<_Tp, M...> >
{
	static_assert(std::is_same<integer_sequence<_Tp, N...>,
			integer_sequence<_Tp, M...> >::value, "extent mismatch!");

	typedef integer_sequence<_Tp, N...> type;
};

}
// namespace _impl

template<typename TOP, typename ... T>
struct extents<nTuple<Expression<TOP, T...> > > : public _impl::extents_helper<
		TOP, traits::extents_t<T>...>::type
{
};

template<typename TV, size_t N, size_t ...M>
struct value_type<nTuple<TV, N, M...> >
{
	typedef traits::value_type_t<TV> type;
};

template<typename TOP, typename ... T>
struct value_type<nTuple<Expression<TOP, T...> > >
{
	typedef traits::result_of_t<TOP(traits::value_type_t<T>...)> type;
};

}  // namespace traits
template<typename T1, typename ...T>
nTuple<T1, 1 + sizeof...(T)> make_nTuple(T1 &&a1, T &&... a)
{
	return std::move(nTuple<T1, 1 + sizeof...(T)>(
			{std::forward<T1>(a1), std::forward<T>(a)...}));
}

template<typename TInts, TInts ...N>
nTuple<TInts, sizeof...(N)> seq2ntuple(integer_sequence<TInts, N...>)
{
	return std::move(nTuple<TInts, sizeof...(N)>({N...}));
}

template<typename TV, size_t N, typename T1>
nTuple<TV, N> append_ntuple(T1 const &v0, TV const &v1)
{
	nTuple<TV, N> res;
	res = v0;
	res[N - 1] = v1;
	return std::move(res);
}

template<typename TV, size_t N, typename T2>
nTuple<TV, N + 1> join_ntuple(nTuple<TV, N> const &left, T2 right)
{
	nTuple<TV, N + 1> res;
	res = left;
	res[N] = right;
	return std::move(res);
}

template<typename T1, size_t N, typename T2, size_t M>
nTuple<T1, N + M> join_ntuple(nTuple<T1, N> const &left, nTuple<T2, M> right)
{
	nTuple<T1, N + M> res;
	for (int i = 0; i < N; ++i)
	{
		res[i] = left[i];
	}
	for (int i = 0; i < M; ++i)
	{
		res[i + N] = right[i];
	}
	return std::move(res);
}

template<typename T, size_t N> using Vector=nTuple<T, N>;

template<typename T, size_t M, size_t N> using Matrix=nTuple<T, M, N>;

template<typename T, size_t ... N> using Tensor=nTuple<T, N...>;

template<typename T>
inline auto determinant(nTuple<T, 3> const &m)
DECL_RET_TYPE(m[0] * m[1] * m[2])

template<typename T>
inline auto determinant(nTuple<T, 4> const &m)
DECL_RET_TYPE(m[0] * m[1] * m[2] * m[3])

template<typename T>
inline auto determinant(
		Matrix<T, 3, 3> const &m)
DECL_RET_TYPE((m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] //
		* m[1][2] * m[2][0] -
		m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1]//
		* m[0][2] - m[1][2] * m[2][1] * m[0][0]//
)

)

template<typename T>
inline auto determinant(
		Matrix<T, 4, 4> const &m)
DECL_RET_TYPE((//
		m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1] * m[3][0]//
				- m[0][3] * m[1][1] * m[2][2] * m[3][0] + m[0][1] * m[1][3]//
				* m[2][2] * m[3][0] +
				m[0][2] * m[1][1] * m[2][3] * m[3][0] - m[0][1]//
				* m[1][2] * m[2][3] * m[3][0] -
				m[0][3] * m[1][2] * m[2][0] * m[3][1]//
						+ m[0][2] * m[1][3] * m[2][0] * m[3][1] + m[0][3] * m[1][0] * m[2][2]//
				* m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1] -
				m[0][2] * m[1][0]//
						* m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1] + m[0][3]//
				* m[1][1] * m[2][0] * m[3][2] -
				m[0][1] * m[1][3] * m[2][0] * m[3][2]//
						- m[0][3] * m[1][0] * m[2][1] * m[3][2] + m[0][0] * m[1][3] * m[2][1]//
				* m[3][2] + m[0][1] * m[1][0] * m[2][3] * m[3][2] -
				m[0][0] * m[1][1]//
						* m[2][3] * m[3][2] - m[0][2] * m[1][1] * m[2][0] * m[3][3] + m[0][1]//
				* m[1][2] * m[2][0] * m[3][3] +
				m[0][2] * m[1][0] * m[2][1] * m[3][3]//
						- m[0][0] * m[1][2] * m[2][1] * m[3][3] - m[0][1] * m[1][0] * m[2][2]//
				* m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3]//
))

template<typename T1, size_t ... N1, typename T2, size_t ... N2>
inline auto cross(nTuple<T1, N1...> const &l, nTuple<T2, N2...> const &r)
-> nTuple<decltype(traits::index(l, 0) * traits::index(r, 0)), 3>
{
	nTuple<decltype(traits::index(l, 0) * traits::index(r, 0)), 3> res = {

			traits::index(l, 1) * traits::index(r, 2)
					- traits::index(l, 2) * traits::index(r, 1),

			traits::index(l, 2) * traits::index(r, 0)
					- traits::index(l, 0) * traits::index(r, 2),

			traits::index(l, 0) * traits::index(r, 1)
					- traits::index(l, 1) * traits::index(r, 0)};
	return std::move(res);
}

//inline nTuple<double, 3> cross(nTuple<double, 3> const &l,
//		nTuple<double, 3> const &r)
//{
//	return std::move(
//			nTuple<double, 3>(
//					{ l[1] * r[2] - l[2] * r[1], l[2] * r[0] - l[0] * r[2], l[0]
//							* r[1] - l[1] * r[0] }));
//}

namespace _impl
{

template<size_t...> struct value_in_range;

template<size_t N, size_t ...DIMS>
struct value_in_range<N, DIMS...>
{
	template<typename T0, typename T1, typename T2>
	static bool eval(T0 const &b, T1 const &e, T2 const &x)
	{

		for (int i = 0; i < N; ++i)
		{
			if (!value_in_range<DIMS...>::eval(b[i], e[i], x[i]))
			{
				return false;
			}
		}
		return true;
	}

};

template<>
struct value_in_range<>
{
	template<typename T0, typename T1, typename T2>
	static bool eval(T0 const &b, T1 const &e, T2 const &x)
	{
		return x >= b && x < e;
	}

};
}  // namespace _impl

template<size_t ... DIMS, typename T0, typename T1, typename T2>
bool value_in_range(T0 const &b, T1 const &e, T2 const &x)
{
	return _impl::value_in_range<DIMS...>::eval(b, e, x);
}

//template<typename T, size_t ...N>
//auto mod(nTuple<T, N...> const & l)
//DECL_RET_TYPE((std::sqrt(std::abs(inner_product(l,l)))))

#define _SP_DEFINE_nTuple_EXPR_BINARY_RIGHT_OPERATOR(_OP_, _NAME_)                                                  \
    template<typename T1,size_t ...N1,typename  T2> \
    constexpr nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>> \
    operator _OP_(nTuple<T1,N1...> const & l,T2 const &r)  \
    {return (nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>>(l,r)) ;}                 \


#define _SP_DEFINE_nTuple_EXPR_BINARY_OPERATOR(_OP_, _NAME_)                                                  \
    template<typename T1,size_t ...N1,typename  T2> \
    constexpr nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>> \
    operator _OP_(nTuple<T1, N1...> const & l,T2 const&r)  \
    {return (nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>>(l,r));}                    \
    \
    template< typename T1,typename T2 ,size_t ...N2> \
    constexpr nTuple<Expression< _impl::_NAME_,T1,nTuple< T2,N2...>>> \
    operator _OP_(T1 const & l, nTuple< T2,N2...>const &r)                    \
    {return (nTuple<Expression< _impl::_NAME_,T1,nTuple< T2,N2...>>>(l,r))  ;}                \
    \
    template< typename T1,size_t ... N1,typename T2 ,size_t ...N2>  \
    constexpr nTuple<Expression< _impl::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>\
    operator _OP_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
    {return (nTuple<Expression< _impl::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>(l,r));}                    \


#define _SP_DEFINE_nTuple_EXPR_UNARY_OPERATOR(_OP_, _NAME_)                           \
        template<typename T,size_t ...N> \
        constexpr nTuple<Expression<_impl::_NAME_,nTuple<T,N...> >> \
        operator _OP_(nTuple<T,N...> const &l)  \
        {return (nTuple<Expression<_impl::_NAME_,nTuple<T,N...> >>(l)) ;}    \


#define _SP_DEFINE_nTuple_EXPR_BINARY_FUNCTION(_NAME_)                                                  \
            template<typename T1,size_t ...N1,typename  T2> \
            constexpr    nTuple<BooleanExpression<_impl::_##_NAME_,nTuple<T1,N1...>,T2>> \
            _NAME_(nTuple<T1,N1...> const & l,T2 const &r)  \
            {return (nTuple<BooleanExpression<_impl::_##_NAME_,nTuple<T1,N1...>,T2>>(l,r));}       \
            \
            template< typename T1,typename T2,size_t ...N2> \
            constexpr    nTuple<Expression< _impl::_##_NAME_,T1,nTuple< T2,N2...>>>\
            _NAME_(T1 const & l, nTuple< T2,N2...>const &r)                    \
            {return (nTuple<Expression< _impl::_##_NAME_,T1,nTuple< T2,N2...>>>(l,r)) ;}       \
            \
            template< typename T1,size_t ... N1,typename T2,size_t  ...N2> \
            constexpr    nTuple<Expression< _impl::_##_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>\
            _NAME_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
            {return (nTuple<Expression< _impl::_##_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>(l,r))  ;}   \


#define _SP_DEFINE_nTuple_EXPR_UNARY_FUNCTION(_NAME_)                           \
        template<typename T,size_t ...N> \
        constexpr nTuple<Expression<_impl::_##_NAME_,nTuple<T,N...>>> \
        _NAME_(nTuple<T,N ...> const &r)  \
        {return (nTuple<Expression<_impl::_##_NAME_,nTuple<T,N...>>>(r));}     \


DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA2(nTuple)

/** @}*/

//////////////////////////////////////////////////////////////
/// @name reduction operation
//////////////////////////////////////////////////////////////
namespace traits
{

template<typename TOP, typename ...T>
struct primary_type<nTuple<BooleanExpression<TOP, T...> > >
{
	typedef bool type;
};
template<typename TOP, typename ...T>
struct pod_type<nTuple<BooleanExpression<TOP, T...> > >
{
	typedef bool type;
};
template<typename TOP, typename ...T>
struct extents<nTuple<BooleanExpression<TOP, T...> > > : public traits::extents_t<
		nTuple<Expression<TOP, T...> > >
{

};
template<typename TOP, typename ...T>
struct value_type<nTuple<BooleanExpression<TOP, T...> > >
{
	typedef bool type;
};
}  // namespace traits

template<typename TOP, typename T>
T const &reduce(TOP const &op, T const &v)
{
	return v;
}

template<typename TOP, typename T, size_t ...N>
traits::value_type_t<nTuple<T, N...>> reduce(TOP const &op,
		nTuple<T, N...> const &v)
{
	static constexpr size_t n = traits::extent<nTuple<T, N...>, 0>::value;

	traits::value_type_t<nTuple<T, N...> > res = reduce(op,
			traits::index(v, 0));
	if (n > 1)
	{
		for (int s = 1; s < n; ++s)
		{
			res = op(res, reduce(op, traits::index(v, s)));
		}
	}
	return res;
}

template<typename TOP, typename ...T>
traits::value_type_t<nTuple<Expression<T...> > > reduce(TOP const &op,
		nTuple<Expression<T...> > const &v)
{
	traits::primary_type_t<nTuple<Expression<T...> > > res = v;

	return reduce(op, res);
}

//template<typename TOP, typename ...Args>
//auto for_each(TOP const & op, Args &&... args)
//-> typename std::enable_if<!(mpl::logical_or<
//		traits::is_ntuple<Args>::value...>::value),void>::type
//{
//	op(std::forward<Args>(args)...);
//}

template<typename TOP, typename ...Args>
void for_each(TOP const &op, integer_sequence<size_t>, Args &&... args)
{
	op(std::forward<Args>(args) ...);
}

template<size_t N, size_t ...M, typename TOP, typename ...Args>
void for_each(TOP const &op, integer_sequence<size_t, N, M...>,
		Args &&... args)
{
	for (size_t s = 0; s < N; ++s)
	{
		for_each(op, integer_sequence<size_t, M...>(),
				traits::index(std::forward<Args>(args), s)...);
	}

}

template<typename TR, typename T, size_t ... N>
auto inner_product(nTuple<T, N...> const &l, TR const &r)
DECL_RET_TYPE ((reduce(_impl::plus(), l * r)))

template<typename TR, typename T, size_t ... N>
auto dot(nTuple<T, N...> const &l, TR const &r)
DECL_RET_TYPE ((inner_product(l, r)))

template<typename T, size_t ... N>
auto normal(nTuple<T, N...> const &l)
DECL_RET_TYPE((std::sqrt((inner_product(l, l)))))

template<typename T>
auto sp_abs(T const &v)
DECL_RET_TYPE((std::abs(v)))

template<typename T, size_t ...N>
auto sp_abs(nTuple<T, N...> const &m)
DECL_RET_TYPE((std::sqrt(std::abs(inner_product(m, m)))))

template<typename ... T>
auto sp_abs(nTuple<Expression<T...> > const &m)
DECL_RET_TYPE((std::sqrt(std::abs(inner_product(m, m)))))

template<typename T> auto mod(T const &v) DECL_RET_TYPE((sp_abs(v)))

template<typename T, size_t ...N>
auto abs(nTuple<T, N...> const &v) DECL_RET_TYPE((sp_abs(v)))

template<typename T, size_t ...N>
inline auto NProduct(nTuple<T, N...> const &v)
DECL_RET_TYPE((reduce(_impl::multiplies(), v)))

template<typename T, size_t ...N>
inline auto NSum(nTuple<T, N...> const &v)
DECL_RET_TYPE((reduce(_impl::plus(), v)))

template<typename TOP, typename ... T>
struct nTuple<BooleanExpression<TOP, T...> > : public nTuple<
		Expression<TOP, T...> >
{
	typedef nTuple<BooleanExpression<TOP, T...>> this_type;

	using nTuple<Expression<TOP, T...>>::at;
	using nTuple<Expression<TOP, T...>>::nTuple;

	operator bool() const
	{
		static constexpr size_t N = mpl::max<size_t,
				traits::extent<T, 0>::value...>::value;

		bool res = static_cast<bool>(at(0));

		if (N > 1)
		{
			for (size_t s = 1; s < N; ++s)
			{
				res = typename _impl::op_traits<TOP>::type()(res,
						static_cast<bool>(at(s)));
			}
		}
		return res;
	}

};

#define _SP_DEFINE_nTuple_EXPR_BINARY_BOOLEAN_OPERATOR(_OP_, _NAME_)                                                  \
    template<typename T1,size_t ...N1,typename  T2> \
    constexpr nTuple<BooleanExpression<_impl::_NAME_,nTuple<T1,N1...>,T2>> \
    operator _OP_(nTuple<T1, N1...> const & l,T2 const&r)  \
    {return (nTuple<BooleanExpression<_impl::_NAME_,nTuple<T1,N1...>,T2>>(l,r));}                    \
    \
    template< typename T1,typename T2 ,size_t ...N2> \
    constexpr nTuple<BooleanExpression< _impl::_NAME_,T1,nTuple< T2,N2...>>> \
    operator _OP_(T1 const & l, nTuple< T2,N2...>const &r)                    \
    {return (nTuple<BooleanExpression< _impl::_NAME_,T1,nTuple< T2,N2...>>>(l,r))  ;}                \
    \
    template< typename T1,size_t ... N1,typename T2 ,size_t ...N2>  \
    constexpr nTuple<BooleanExpression< _impl::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>\
    operator _OP_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
    {return (nTuple<BooleanExpression< _impl::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>(l,r));}                    \


#define _SP_DEFINE_nTuple_EXPR_UNARY_BOOLEAN_OPERATOR(_OP_, _NAME_)                           \
        template<typename T,size_t ...N> \
        constexpr nTuple<BooleanExpression<_impl::_NAME_,nTuple<T,N...> >> \
        operator _OP_(nTuple<T,N...> const &l)  \
        {return (nTuple<BooleanExpression<_impl::_NAME_,nTuple<T,N...> >>(l)) ;}    \


DEFINE_EXPRESSOPM_TEMPLATE_BOOLEAN_ALGEBRA2(nTuple)

}
//namespace simpla

namespace std
{

template<typename T, size_t N, size_t ... M>
void swap(simpla::nTuple<T, N, M...> &l, simpla::nTuple<T, N, M...> &r)
{
	for (size_t s = 0; s < N; ++s)
	{
		swap(simpla::traits::index(l, s), simpla::traits::index(r, s));
	}
}

template<typename T, size_t N, size_t ... M>
void swap(simpla::nTuple<T, N, M...> &l,
		simpla::traits::pod_type_t<simpla::nTuple<T, N, M...>> &r)
{

	for (size_t s = 0; s < N; ++s)
	{
		swap(simpla::traits::index(l, s), simpla::traits::index(r, s));
	}
}

}  // namespace std

#endif  // CORE_GTL_NTUPLE_H_
