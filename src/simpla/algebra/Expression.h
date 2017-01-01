/**
 * @file expression_template.h
 *
 *  Created on: 2014-9-26
 *      Author: salmon
 */

#ifndef EXPRESSION_TEMPLATE_H_
#define EXPRESSION_TEMPLATE_H_

#include <cmath>
#include <limits>
#include <type_traits>
#include <complex>
#include <tuple>
#include <cmath>
#include <simpla/mpl/type_traits.h>
#include <simpla/mpl/integer_sequence.h>
#include "Algebra.h"

namespace simpla { namespace algebra
{
namespace declare { template<typename ...> struct Expression; }

namespace traits
{


template<typename TOP, typename ...Others>
struct is_scalar<declare::Expression<TOP, Others...> > : public is_scalar<Others...> {};

template<typename TOP, typename ...Others>
struct is_nTuple<declare::Expression<TOP, Others...> > : public is_nTuple<Others...> {};

template<typename TOP, typename ...Others>
struct is_field<declare::Expression<TOP, Others...> > : public is_field<Others...> {};

template<typename TOP, typename ...Others>
struct is_array<declare::Expression<TOP, Others...> > : public is_array<Others...> {};

template<typename TOP, typename T0, typename ... T>
struct iform<declare::Expression<TOP, T0, T...> > : public iform<T0> {};

template<typename TOP, typename ...Others>
struct value_type<declare::Expression<TOP, Others...> >
{
    typedef std::result_of_t<TOP(typename value_type<Others>::type ...)> type;
};


}//namespace traits


namespace declare
{

/**
 *  @ingroup calculus
 *  @addtogroup expression_template  Expression Template
 *  @{
 */
template<typename ...> class Expression;

template<typename ...> struct BooleanExpression;

template<typename ...> struct AssignmentExpression;

}
namespace calculus
{
template<typename ...> struct eval_expr_as;
};

namespace declare
{

template<typename TOP, typename ...Args>
struct Expression<TOP, Args...>
{
    typedef Expression<TOP, Args...> this_type;

    typename std::tuple<traits::reference_t<Args> ...> m_args_;

    TOP m_op_;

    Expression(this_type const &that) : m_args_(that.m_args_) {}

    Expression(this_type &&that) : m_args_(that.m_args_) {}

    Expression(Args &... args) : m_args_(args...) {}

    virtual ~Expression() {}

//    template<typename T> operator T() const { return calculus::calculator<this_type>::template cast_as<T>(*this); }

};

template<typename TOP, typename TL, typename TR>
struct BooleanExpression<TOP, TL, TR> : public Expression<TOP, TL, TR>
{
    typedef Expression<TOP, TL, TR> base_type;

    using Expression<TOP, TL, TR>::Expression;

    operator bool() const { return false; }

    BooleanExpression(TL const &l, TR const &r) : base_type(l, r) {}
};

template<typename TOP, typename TL>
struct BooleanExpression<TOP, TL> : public Expression<TOP, TL>
{
    typedef Expression<TOP, TL> base_type;

    using Expression<TOP, TL>::Expression;

    BooleanExpression(TL const &l) : base_type(l) {}

    operator bool() const { return false; }
};
//
//template<typename TOP, typename TL, typename TR>
//struct AssignmentExpression<TOP, TL, TR>
//{
//    typedef AssignmentExpression<TOP, TL, TR> this_type;
//    TL &lhs;
//    typename traits::reference<TR>::type rhs;
//    TOP op_;
//
//    AssignmentExpression(this_type const &that) : lhs(that.lhs), rhs(that.rhs), op_(that.op_) {}
//
//    AssignmentExpression(this_type &&that) : lhs(that.lhs), rhs(that.rhs), op_(that.op_) {}
//
//    AssignmentExpression(TL &l, TR const &r) : lhs(l), rhs(r), op_() {}
//
//    AssignmentExpression(TOP op, TL &l, TR const &r) : lhs(l), rhs(r), op_(op) {}
//
//    virtual   ~AssignmentExpression() {}
//
//    template<typename IndexType>
//    inline auto operator[](IndexType const &s) const
//    DECL_RET_TYPE (((op_(traits::get_value(lhs, s), traits::get_value(rhs, s)))))
//
//};

}
}}  // namespace simpla::algebra::declare

#endif /* EXPRESSION_TEMPLATE_H_ */
