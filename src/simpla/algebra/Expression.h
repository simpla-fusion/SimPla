/**
 * @file expression_template.h
 *
 *  Created on: 2014-9-26
 *      Author: salmon
 */

#ifndef EXPRESSION_TEMPLATE_H_
#define EXPRESSION_TEMPLATE_H_

#include <simpla/mpl/integer_sequence.h>
#include <simpla/mpl/type_traits.h>
#include <cmath>
#include <cmath>
#include <complex>
#include <limits>
#include <tuple>
#include <type_traits>
#include "Algebra.h"
#include "Calculus.h"

namespace simpla {
namespace algebra {

namespace declare {
template <typename...>
struct Expression;
}
namespace calculus {

template <typename...>
struct expr_parser;

template <typename TRes, typename TR>
struct expr_parser<TRes, TR> {
    static TRes eval(TR const &expr) { return static_cast<TRes>(expr); };
};
template <typename TRes, typename TOP, typename... Args>
struct expr_parser<TRes, declare::Expression<TOP, Args...>> {
    template <size_type... index>
    static decltype(auto) _invoke_helper(declare::Expression<TOP, Args...> const &expr,
                                         index_sequence<index...>) {
        return expr.m_op_(
            expr_parser<TRes, std::remove_cv_t<Args>>::eval(std::get<index>(expr.m_args_))...);
    }

    static TRes eval(declare::Expression<TOP, Args...> const &expr) {
        return _invoke_helper(expr, index_sequence_for<Args...>());
    };
};
}
namespace traits {

template <typename TOP, typename... Args>
struct is_scalar<declare::Expression<TOP, Args...>> : public is_scalar<Args...> {};

template <typename TOP, typename... Args>
struct is_nTuple<declare::Expression<TOP, Args...>> : public is_nTuple<Args...> {};

template <typename TOP, typename... Args>
struct is_field<declare::Expression<TOP, Args...>> : public is_field<Args...> {};

template <typename TOP, typename... Args>
struct is_array<declare::Expression<TOP, Args...>> : public is_array<Args...> {};

template <typename TOP, typename... T>
struct iform<declare::Expression<TOP, T...>>
    : public int_const<seq_max<int, iform<T>::value...>::value> {};

template <typename TOP, typename... T>
struct extent<declare::Expression<TOP, T...>>
    : public int_const<seq_max<int, extent<T>::value...>::value> {};

template <typename TOP, typename TL>
struct value_type<declare::Expression<TOP, TL>> {
    typedef std::result_of_t<TOP(value_type_t<TL>)> type;
};

template <typename TOP, typename TL, typename TR>
struct value_type<declare::Expression<TOP, TL, TR>> {
    typedef std::result_of_t<TOP(value_type_t<TL>, value_type_t<TR>)> type;
};
template <typename TOP, typename... T>
struct value_type<declare::Expression<TOP, T...>> {
    typedef std::result_of_t<TOP(value_type_t<T>...)> type;
};
}  // namespace traits

namespace declare {

/**
 *  @ingroup calculus
 *  @addtogroup expression_template  Expression Template
 *  @{
 */
template <typename...>
class Expression;

template <typename...>
struct BooleanExpression;

template <typename...>
struct AssignmentExpression;

template <typename TOP, typename... Args>
struct Expression<TOP, Args...> {
    typedef Expression<TOP, Args...> this_type;

    typename std::tuple<traits::reference_t<Args>...> m_args_;
    typedef std::true_type is_expression;
    TOP m_op_;

    Expression(this_type const &that) : m_args_(that.m_args_) {}

    Expression(this_type &&that) : m_args_(that.m_args_) {}

    Expression(Args &... args) : m_args_(args...) {}

    virtual ~Expression() {}

    template <typename T>
    operator T() const {
        return calculus::expr_parser<T, this_type>::eval(*this);
    }
};

template <typename TOP, typename TL, typename TR>
struct BooleanExpression<TOP, TL, TR> : public Expression<TOP, TL, TR> {
    typedef Expression<TOP, TL, TR> base_type;

    using Expression<TOP, TL, TR>::Expression;

    operator bool() const { return false; }

    BooleanExpression(TL const &l, TR const &r) : base_type(l, r) {}
};

template <typename TOP, typename TL>
struct BooleanExpression<TOP, TL> : public Expression<TOP, TL> {
    typedef Expression<TOP, TL> base_type;

    using Expression<TOP, TL>::Expression;

    BooleanExpression(TL const &l) : base_type(l) {}

    operator bool() const { return false; }
};
//
// template<typename TOP, typename TL, typename TR>
// struct AssignmentExpression<TOP, TL, TR>
//{
//    typedef AssignmentExpression<TOP, TL, TR> this_type;
//    TL &lhs;
//    typename traits::reference<TR>::type rhs;
//    TOP op_;
//
//    AssignmentExpression(this_type const &that) : lhs(that.lhs),
//    rhs(that.rhs), op_(that.op_) {}
//
//    AssignmentExpression(this_type &&that) : lhs(that.lhs), rhs(that.rhs),
//    op_(that.op_) {}
//
//    AssignmentExpression(TL &l, TR const &r) : lhs(l), rhs(r), op_() {}
//
//    AssignmentExpression(TOP op, TL &l, TR const &r) : lhs(l), rhs(r), op_(op)
//    {}
//
//    virtual   ~AssignmentExpression() {}
//
//    template<typename IndexType>
//    inline auto operator[](IndexType const &s) const
//    AUTO_RETURN (((op_(traits::get_value(lhs, s), traits::get_value(rhs,
//    s)))))
//
//};
}  // namespace declare

}  // namespace algebra
}  // namespace simpla

#endif /* EXPRESSION_TEMPLATE_H_ */
