/**
 *@file ntuple_reduce.h
 *
 *  Created on: 2015年6月21日
 *      Author: salmon
 */

#ifndef CORE_GTL_NTUPLE_EXT1_H_
#define CORE_GTL_NTUPLE_EXT1_H_
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "macro.h"
#include "type_traits.h"
#include "integer_sequence.h"
#include "expression_template.h"
#include "mpl.h"
namespace simpla
{
template<typename, size_t...>struct nTuple;
template<typename ...>class Expression;
template<typename ...>class BooleanExpression;

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
T const & reduce(TOP const & op, T const & v)
{
	return v;
}
template<typename TOP, typename T, size_t ...N>
traits::value_type_t<nTuple<T, N...>> reduce(TOP const & op,
		nTuple<T, N...> const & v)
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
traits::value_type_t<nTuple<Expression<T...> > > reduce(TOP const & op,
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
void for_each(TOP const & op, integer_sequence<size_t>, Args &&... args)
{
	op(std::forward<Args>(args) ...);
}
template<size_t N, size_t ...M, typename TOP, typename ...Args>
void for_each(TOP const & op, integer_sequence<size_t, N, M...>,
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
DECL_RET_TYPE ((reduce( _impl::plus(), l * r)) )

template<typename TR, typename T, size_t ... N>
auto dot(nTuple<T, N...> const &l, TR const &r)
DECL_RET_TYPE ((inner_product( l, r)))

template<typename T, size_t ... N>
auto normal(nTuple<T, N...> const &l)
DECL_RET_TYPE((std::sqrt((inner_product( l , l)))))

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
DECL_RET_TYPE(( reduce( _impl::multiplies(), v)))

template<typename T, size_t ...N>
inline auto NSum(nTuple<T, N...> const &v)
DECL_RET_TYPE(( reduce( _impl::plus(), v)))

template<typename TOP, typename ... T>
struct nTuple<BooleanExpression<TOP, T...>> : public Expression<TOP, T...>
{
	typedef nTuple<BooleanExpression<TOP, T...>> this_type;

	using Expression<TOP, T...>::m_op_;
	using Expression<TOP, T...>::args;
	using Expression<TOP, T...>::Expression;

	operator bool() const
	{
		return reduce(_impl::logical_and(), *this);
//		return mpl::seq_reduce(traits::extents_t<this_type>(),
//				typename _impl::op_traits<TOP>::reduction_op(), *this);
	}

};
template<typename ... T>
struct nTuple<BooleanExpression<_impl::not_equal_to, T...>> : public Expression<
		_impl::not_equal_to, T...>
{
	typedef nTuple<BooleanExpression<_impl::not_equal_to, T...>> this_type;

	using Expression<_impl::not_equal_to, T...>::m_op_;
	using Expression<_impl::not_equal_to, T...>::args;
	using Expression<_impl::not_equal_to, T...>::Expression;

	operator bool() const
	{
		return reduce(_impl::logical_or(), *this);
//		return mpl::seq_reduce(traits::extents_t<this_type>(),
//				typename _impl::op_traits<TOP>::reduction_op(), *this);
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

}  // namespace simpla

#endif /* CORE_GTL_NTUPLE_EXT1_H_ */
