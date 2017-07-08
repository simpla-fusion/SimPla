/**
 * @file expression_template.h
 *
 *  Created on: 2014-9-26
 *      Author: salmon
 */

#ifndef EXPRESSION_TEMPLATE_H_
#define EXPRESSION_TEMPLATE_H_

#include <simpla/utilities/integer_sequence.h>
#include <simpla/utilities/type_traits.h>
#include <cmath>
#include <cmath>
#include <complex>
#include <limits>
#include <tuple>
#include <type_traits>
#include "simpla/algebra/Algebra.h"
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
    static auto _invoke_helper(declare::Expression<TOP, Args...> const &expr, index_sequence<index...>) {
        return expr.m_op_(expr_parser<TRes, std::remove_cv_t<Args>>::eval(std::get<index>(expr.m_args_))...);
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
struct iform<declare::Expression<TOP, T...>> : public int_const<seq_max<int, iform<T>::value...>::value> {};
template <typename TOP, typename T>
struct iform<declare::Expression<TOP, T>> : public int_const<iform<T>::value> {};

template <typename TOP, typename... T>
struct extent<declare::Expression<TOP, T...>> : public int_const<seq_max<int, extent<T>::value...>::value> {};

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
    typedef std::false_type prefer_pass_by_reference;
    typedef std::true_type prefer_pass_by_value;

    TOP m_op_;

    Expression(this_type const &that) : m_args_(that.m_args_) {}

    Expression(this_type &&that) noexcept : m_args_(that.m_args_) {}

    explicit Expression(Args &... args) noexcept : m_args_(args...) {}

    virtual ~Expression() = default;

    this_type &operator=(this_type const &) = delete;
    this_type &operator=(this_type &&) = delete;

    template <typename T>
    explicit operator T() const {
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

    explicit BooleanExpression(TL const &l) : base_type(l) {}

    operator bool() const { return false; }
};
//
// template<typename TOP, typename TL, typename TR>
// struct AssignmentExpression<TOP, TL, TR>
//{
//    typedef AssignmentExpression<TOP, TL, TR> this_type;
//    TL &lhs;
//    typename traits::reference<TR>::value_type_info rhs;
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
//    AUTO_RETURN (((op_(traits::GetValue(lhs, s), traits::GetValue(rhs,
//    s)))))
//
//};

template <typename...>
struct Expression;
template <typename...>
struct BooleanExpression;
template <typename...>
struct AssignmentExpression;

#define _SP_DEFINE_EXPR_BINARY_OPERATOR(_OP_, _NAME_)                  \
    template <typename T1, typename T2>                                \
    auto operator _OP_(T1 &l, T2 &r) {                                 \
        return ((Expression<tags::_NAME_, T1, T2>(l, r)));             \
    }                                                                  \
    template <typename T1, typename T2>                                \
    auto operator _OP_(T1 const &l, T2 const &r) {                     \
        return ((Expression<tags::_NAME_, const T1, const T2>(l, r))); \
    }

#define _SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR(_OP_, _NAME_)            \
    template <typename T1, typename T2>                                \
    auto operator _OP_(T1 &l, T2 &r) {                                 \
        return ((Expression<tags::_NAME_, T1, T2>(l, r)));             \
    }                                                                  \
    template <typename T1, typename T2>                                \
    auto operator _OP_(T1 const &l, T2 const &r) {                     \
        return ((Expression<tags::_NAME_, const T1, const T2>(l, r))); \
    }

#define _SP_DEFINE_EXPR_UNARY_OPERATOR(_OP_, _NAME_)      \
    template <typename T1>                                \
    auto operator _OP_(T1 &l) {                           \
        return ((Expression<tags::_NAME_, T1>(l)));       \
    }                                                     \
    template <typename T1>                                \
    auto operator _OP_(T1 const &l) {                     \
        return ((Expression<tags::_NAME_, const T1>(l))); \
    }

#define _SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(_OP_, _NAME_)                 \
    template <typename T1, typename T2>                                       \
    auto operator _OP_(T1 &l, T2 &r) {                                        \
        return ((BooleanExpression<tags::_NAME_, T1, T2>(l, r)));             \
    }                                                                         \
    template <typename T1, typename T2>                                       \
    auto operator _OP_(T1 const &l, T2 const &r) {                            \
        return ((BooleanExpression<tags::_NAME_, const T1, const T2>(l, r))); \
    }

#define _SP_DEFINE_EXPR_UNARY_BOOLEAN_OPERATOR(_OP_, _NAME_)     \
    template <typename T1>                                       \
    auto operator _OP_(T1 &l) {                                  \
        return ((BooleanExpression<tags::_NAME_, T1>(l)));       \
    }                                                            \
    template <typename T1>                                       \
    auto operator _OP_(T1 const &l) {                            \
        return ((BooleanExpression<tags::_NAME_, const T1>(l))); \
    }

#define _SP_DEFINE_EXPR_BINARY_FUNCTION(_NAME_)                           \
    template <typename T1, typename T2>                                   \
    auto _NAME_(T1 &l, T2 &r) {                                           \
        return ((Expression<tags::_##_NAME_, T1, T2>(l, r)));             \
    }                                                                     \
    template <typename T1, typename T2>                                   \
    auto _NAME_(T1 const &l, T2 const &r) {                               \
        return ((Expression<tags::_##_NAME_, const T1, const T2>(l, r))); \
    }

#define _SP_DEFINE_EXPR_UNARY_FUNCTION(_NAME_)               \
    template <typename T1>                                   \
    auto _NAME_(T1 &l) {                                     \
        return ((Expression<tags::_##_NAME_, T1>(l)));       \
    }                                                        \
    template <typename T1>                                   \
    auto _NAME_(T1 const &l) {                               \
        return ((Expression<tags::_##_NAME_, const T1>(l))); \
    }

_SP_DEFINE_EXPR_BINARY_OPERATOR(+, plus)
_SP_DEFINE_EXPR_BINARY_OPERATOR(-, minus)
_SP_DEFINE_EXPR_BINARY_OPERATOR(*, multiplies)
_SP_DEFINE_EXPR_BINARY_OPERATOR(/, divides)
_SP_DEFINE_EXPR_BINARY_OPERATOR(%, modulus)
_SP_DEFINE_EXPR_BINARY_OPERATOR (^, bitwise_xor)
_SP_DEFINE_EXPR_BINARY_OPERATOR(&, bitwise_and)
_SP_DEFINE_EXPR_BINARY_OPERATOR(|, bitwise_or)
_SP_DEFINE_EXPR_UNARY_OPERATOR(~, bitwise_not)
_SP_DEFINE_EXPR_UNARY_OPERATOR(+, unary_plus)
_SP_DEFINE_EXPR_UNARY_OPERATOR(-, negate)
_SP_DEFINE_EXPR_UNARY_FUNCTION(cos)
_SP_DEFINE_EXPR_UNARY_FUNCTION(acos)
_SP_DEFINE_EXPR_UNARY_FUNCTION(cosh)
_SP_DEFINE_EXPR_UNARY_FUNCTION(sin)
_SP_DEFINE_EXPR_UNARY_FUNCTION(asin)
_SP_DEFINE_EXPR_UNARY_FUNCTION(sinh)
_SP_DEFINE_EXPR_UNARY_FUNCTION(tan)
_SP_DEFINE_EXPR_UNARY_FUNCTION(tanh)
_SP_DEFINE_EXPR_UNARY_FUNCTION(atan)
_SP_DEFINE_EXPR_UNARY_FUNCTION(exp)
_SP_DEFINE_EXPR_UNARY_FUNCTION(log)
_SP_DEFINE_EXPR_UNARY_FUNCTION(log10)
_SP_DEFINE_EXPR_UNARY_FUNCTION(sqrt)
_SP_DEFINE_EXPR_BINARY_FUNCTION(atan2)
_SP_DEFINE_EXPR_BINARY_FUNCTION(pow)
_SP_DEFINE_EXPR_UNARY_BOOLEAN_OPERATOR(!, logical_not)
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(&&, logical_and)
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(||, logical_or)
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(!=, not_equal_to)
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(==, equal_to)
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(<, less)
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(>, greater)
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(<=, less_equal)
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(>=, greater_equal)

//_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(real)                                          \
//_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(imag)                                          \

template <typename T1>
auto operator<<(T1 const &l, unsigned int r) {
    return ((Expression<tags::shift_left, const T1, unsigned int>(l, r)));
}
template <typename T1>
auto operator>>(T1 const &l, unsigned int r) {
    return ((Expression<tags::shift_right, const T1, unsigned int>(l, r)));
}

#undef _SP_DEFINE_EXPR_BINARY_OPERATOR
#undef _SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR
#undef _SP_DEFINE_EXPR_UNARY_OPERATOR
#undef _SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR
#undef _SP_DEFINE_EXPR_UNARY_BOOLEAN_OPERATOR
#undef _SP_DEFINE_EXPR_BINARY_FUNCTION
#undef _SP_DEFINE_EXPR_UNARY_FUNCTION

#define _SP_DEFINE_COMPOUND_OP(_OP_)               \
    template <typename TL, typename TR>            \
    TL &operator _OP_##=(TL &lhs, TR const &rhs) { \
        lhs = lhs _OP_ rhs;                        \
        return lhs;                                \
    }

_SP_DEFINE_COMPOUND_OP(+)
_SP_DEFINE_COMPOUND_OP(-)
_SP_DEFINE_COMPOUND_OP(*)
_SP_DEFINE_COMPOUND_OP(/)
_SP_DEFINE_COMPOUND_OP(%)
_SP_DEFINE_COMPOUND_OP(&)
_SP_DEFINE_COMPOUND_OP(|)

#undef _SP_DEFINE_COMPOUND_OP
}  // namespace declare
}  // namespace algebra
}  // namespace simpla

#endif /* EXPRESSION_TEMPLATE_H_ */
