/**
 * @file field_expression.h
 *
 *  Created on: 2015-1-30
 *      Author: salmon
 */

#ifndef FIELD_EXPRESSION_H_
#define FIELD_EXPRESSION_H_

#include <stddef.h>
#include <cstdbool>
#include <type_traits>

#include <simpla/mpl/type_traits.h>
#include <simpla/algebra/Expression.h>
#include <simpla/mesh/MeshCommon.h>
#include "simpla/algebra/Calculus.h"

namespace simpla {
/** @addtogroup field Field
 *  @{
 */
template<typename ...> struct Field;
namespace ct=algebra::tags;

/// @name  Field Expression
/// @{
template<typename ...> class Expression;

template<typename ...> class BooleanExpression;

namespace traits
{

template<typename ...> struct value_type;
template<typename Q> struct value_type<Field<Q> > { typedef typename value_type<Q>::type type; };

template<typename ...> struct iform;
template<typename Q> struct iform<Field<Q> > : public typename iform<Q>::type {};


template<typename> struct field_value_type;
template<typename TOP, typename ...T>
struct field_value_type<simpla::BooleanExpression<TOP, T...> > { typedef bool type; };
}//namespace  traits


DEFINE_EXPRESSION_TEMPLATE_BASIC_ALGEBRA(Field)

#define SP_DEF_BINOPField_NTUPLE(_OP_, _NAME_)                                                 \
template<typename ...T1, typename T2, size_t ... N>                                            \
Expression<_impl::_NAME_, Field<T1...>, nTuple<T2, N...> >  operator _OP_(              \
        Field<T1...> const & l, nTuple<T2, N...> const &r)                                    \
{return (Expression<_impl::_NAME_, Field<T1...>, nTuple<T2, N...> >  (l, r));}         \
template<typename T1, size_t ... N, typename ...T2>                                            \
 Expression<_impl::_NAME_, nTuple<T1, N...>, Field<T2...> >   operator _OP_(              \
        nTuple<T1, N...> const & l, Field< T2...>const &r)                                    \
{    return ( Expression< _impl::_NAME_,T1,Field< T2...>> (l,r));}                       \


SP_DEF_BINOPField_NTUPLE(+, plus)

SP_DEF_BINOPField_NTUPLE(-, minus)

SP_DEF_BINOPField_NTUPLE(*, multiplies)

SP_DEF_BINOPField_NTUPLE(/, divides)

SP_DEF_BINOPField_NTUPLE(%, modulus)

SP_DEF_BINOPField_NTUPLE(^, bitwise_xor)

SP_DEF_BINOPField_NTUPLE(&, bitwise_and)

SP_DEF_BINOPField_NTUPLE(|, bitwise_or)

#undef SP_DEF_BINOPField_NTUPLE

/** @} */



template<typename ...> struct Field;


}  // namespace simpla

#endif /* FIELD_EXPRESSION_H_ */
