/**
 *@file ntuple_boolean_expr.h
 *
 *  Created on: 2015年6月21日
 *      Author: salmon
 */

#ifndef CORE_GTL_NTUPLE_BOOLEAN_EXPR_H_
#define CORE_GTL_NTUPLE_BOOLEAN_EXPR_H_
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "macro.h"
#include "type_traits.h"
#include "integer_sequence.h"
#include "expression_template.h"
namespace simpla
{
template<typename, size_t...>struct nTuple;
template<typename ...>class Expression;
template<typename ...>class BooleanExpression;

template<typename TOP, typename ... T>
struct nTuple<BooleanExpression<TOP, T...>> : public Expression<TOP, T...>
{
	typedef nTuple<BooleanExpression<TOP, T...>> this_type;

	using Expression<T...>::m_op_;
	using Expression<T...>::args;
	using Expression<T...>::Expression;

	operator bool() const
	{
		return seq_reduce(traits::extents_t<this_type>(),
				typename _impl::op_traits<TOP>::reduction_op(), *this);
	}

};

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

#endif /* CORE_GTL_NTUPLE_BOOLEAN_EXPR_H_ */
