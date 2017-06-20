//
// Created by salmon on 17-4-28.
//

#ifndef SIMPLA_EXPRESSIONTEMPLATE_H
#define SIMPLA_EXPRESSIONTEMPLATE_H

#include <type_traits>
#include <utility>
#include "../../../../../../../pkg/gcc/7.1.0/include/c++/7.1.0/type_traits"
#include "integer_sequence.h"
#include "type_traits.h"
namespace simpla {
/**
*  @ingroup calculus
*  @addtogroup expression_template  Expression Template
*  @{
*/
template <typename...>
class Expression;

template <typename...>
struct AssignmentExpression;

namespace tags {
struct _assign {
    template <typename TL, typename TR>
    void operator()(TL &lhs, TR const &rhs) const {
        lhs = rhs;
    }
};
}
namespace traits {

template <typename TExpr>
struct is_expression : public false_type {};

template <typename... T>
struct is_expression<Expression<T...>> : public true_type {};

template <typename TOP, typename... Args>
struct rank<Expression<TOP, Args...>> : public seq_max<int_sequence<rank<Args>::value...>> {};

template <int N, typename TOP, typename... Args>
struct extent<Expression<TOP, Args...>, N> : public seq_max<int_sequence<extent<Args, N>::value...>> {};

template <typename...>
struct expr_parser;

template <typename TRes, typename TR>
struct expr_parser<TRes, TR> {
    static TRes eval(TR const &expr) { return static_cast<TRes>(expr); };
};
template <typename TRes, typename TOP, typename... Args>
struct expr_parser<TRes, Expression<TOP, Args...>> {
    template <size_type... index>
    static auto _invoke_helper(Expression<TOP, Args...> const &expr, index_sequence<index...>) {
        return expr.m_op_(expr_parser<TRes, std::remove_cv_t<Args>>::eval(std::get<index>(expr.m_args_))...);
    }

    static auto eval(Expression<TOP, Args...> const &expr) {
        return _invoke_helper(expr, index_sequence_for<Args...>());
    };
};

template <typename T>
struct getValue_s {
    static T const &eval(T const &expr, int n) { return expr; }
};

template <typename T>
struct getValue_s<T *> {
    static auto &eval(T *expr, int n) { return expr[n]; }
};

template <typename T>
auto getValue(T const &expr, int n) {
    return getValue_s<T>::eval(expr, n);
}

template <typename TOP, typename... Args>
struct getValue_s<Expression<TOP, Args...>> {
    template <int... index>
    static auto get(Expression<TOP, Args...> const &expr, int n, index_sequence<index...>) {
        return expr.m_op_(getValue(std::get<index>(expr.m_args_), n)...);
    }
    static auto eval(Expression<TOP, Args...> const &expr, int n) {
        return get(expr, n, index_sequence_for<Args...>());
    }
};

template <typename TReduction, typename TExpr>
auto reduction(TExpr const &expr, ENABLE_IF((rank<TExpr>::value == 0))) {
    return expr;
}

template <typename TReduction, typename TExpr>
auto reduction(TExpr const &expr, ENABLE_IF((rank<TExpr>::value > 0))) {
    auto res = reduction<TReduction>(getValue(expr, 0));
    int n = extent<TExpr>::value;
    for (int s = 1; s < n; ++s) { res = TReduction::eval(res, reduction<TReduction>(getValue(expr, s))); }

    return res;
}

template <typename TReduction, typename TOP, typename... Args>
auto reduction(Expression<TOP, Args...> const &expr, ENABLE_IF((rank<Expression<TOP, Args...>>::value > 0))) {
    auto res = reduction<TReduction>(getValue(expr, 0));
    int n = extent<TExpr>::value;
    for (int s = 1; s < n; ++s) { res = TReduction::eval(res, reduction<TReduction>(getValue(expr, s))); }

    return res;
}
}

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
    template <typename... U>
    explicit Expression(U &&... args) noexcept : m_args_(std::forward<U>(args)...) {}

    virtual ~Expression() = default;

    void swap(this_type &other) { m_args_.swap(other.m_args_); }
    this_type &operator=(this_type const &other) {
        this_type(other).swap(*this);
        return *this;
    };
    this_type &operator=(this_type &&other) {
        this_type(other).swap(*this);
        return *this;
    };

    template <typename T>
    explicit operator T() const {
        //        return traits::expr_parser<T, this_type>::eval(*this);
        return traits::getValue(*this, 0);
    }
};

#define _SP_DEFINE_EXPR_BINARY_OPERATOR(_OP_, _NAME_)                                    \
    namespace tags {                                                                     \
    struct _NAME_ {                                                                      \
        template <typename TL, typename TR>                                              \
        static constexpr auto eval(TL const &l, TR const &r) {                           \
            return ((l _OP_ r));                                                         \
        }                                                                                \
        template <typename TL, typename TR>                                              \
        constexpr auto operator()(TL const &l, TR const &r) const {                      \
            return ((l _OP_ r));                                                         \
        }                                                                                \
    };                                                                                   \
    }                                                                                    \
    template <typename... TL, typename TR>                                               \
    auto operator _OP_(Expression<TL...> const &lhs, TR const &rhs) {                    \
        return Expression<tags::_NAME_, Expression<TL...>, TR>(lhs, rhs);                \
    };                                                                                   \
    template <typename TL, typename... TR>                                               \
    auto operator _OP_(TL const &lhs, Expression<TR...> const &rhs) {                    \
        return Expression<tags::_NAME_, TL, Expression<TR...>>(lhs, rhs);                \
    };                                                                                   \
    template <typename... TL, typename... TR>                                            \
    auto operator _OP_(Expression<TL...> const &lhs, Expression<TR...> const &rhs) {     \
        return Expression<tags::_NAME_, Expression<TL...>, Expression<TR...>>(lhs, rhs); \
    };

#define _SP_DEFINE_EXPR_UNARY_OPERATOR(_OP_, _NAME_)            \
    namespace tags {                                            \
    struct _NAME_ {                                             \
        template <typename TL>                                  \
        static constexpr auto eval(TL const &l) {               \
            return (_OP_(l));                                   \
        }                                                       \
        template <typename TL>                                  \
        constexpr auto operator()(TL const &l) const {          \
            return _OP_(l);                                     \
        }                                                       \
    };                                                          \
    }                                                           \
    template <typename... T>                                    \
    auto operator _OP_(Expression<T...> const &lhs) {           \
        return Expression<tags::_NAME_, Expression<T...>>(lhs); \
    }

#define _SP_DEFINE_EXPR_BINARY_FUNCTION(_NAME_)                                          \
    namespace tags {                                                                     \
    struct _NAME_ {                                                                      \
        template <typename TL, typename TR>                                              \
        static constexpr auto eval(TL const &l, TR const &r) {                           \
            return (_NAME_(l, r));                                                       \
        }                                                                                \
        template <typename TL, typename TR>                                              \
        constexpr auto operator()(TL const &l, TR const &r) const {                      \
            return (_NAME_(l, r));                                                       \
        }                                                                                \
    };                                                                                   \
    }                                                                                    \
    template <typename... TL, typename TR>                                               \
    auto _NAME_(Expression<TL...> const &lhs, TR const &rhs) {                           \
        return Expression<tags::_NAME_, Expression<TL...>, TR>(lhs, rhs);                \
    };                                                                                   \
    template <typename TL, typename... TR>                                               \
    auto _NAME_(TL const &lhs, Expression<TR...> const &rhs) {                           \
        return Expression<tags::_NAME_, TL, Expression<TR...>>(lhs, rhs);                \
    };                                                                                   \
    template <typename... TL, typename... TR>                                            \
    auto _NAME_(Expression<TL...> const &lhs, Expression<TR...> const &rhs) {            \
        return Expression<tags::_NAME_, Expression<TL...>, Expression<TR...>>(lhs, rhs); \
    };

#define _SP_DEFINE_EXPR_UNARY_FUNCTION(_NAME_)                  \
    namespace tags {                                            \
    struct _NAME_ {                                             \
        template <typename TL>                                  \
        static constexpr auto eval(TL &l) {                     \
            return (_NAME_(l));                                 \
        }                                                       \
        template <typename TL>                                  \
        constexpr auto operator()(TL &l) const {                \
            return _NAME_(l);                                   \
        }                                                       \
    };                                                          \
    }                                                           \
    template <typename... T>                                    \
    auto _NAME_(Expression<T...> const &lhs) {                  \
        return Expression<tags::_NAME_, Expression<T...>>(lhs); \
    }

_SP_DEFINE_EXPR_BINARY_OPERATOR(+, addition)
_SP_DEFINE_EXPR_BINARY_OPERATOR(-, subtraction)
_SP_DEFINE_EXPR_BINARY_OPERATOR(*, multiplication)
_SP_DEFINE_EXPR_BINARY_OPERATOR(/, division)
_SP_DEFINE_EXPR_BINARY_OPERATOR(%, modulo)

_SP_DEFINE_EXPR_UNARY_OPERATOR(~, bitwise_not)
_SP_DEFINE_EXPR_BINARY_OPERATOR (^, bitwise_xor)
_SP_DEFINE_EXPR_BINARY_OPERATOR(&, bitwise_and)
_SP_DEFINE_EXPR_BINARY_OPERATOR(|, bitwise_or)
_SP_DEFINE_EXPR_BINARY_OPERATOR(<<, bitwise_left_shift)
_SP_DEFINE_EXPR_BINARY_OPERATOR(>>, bitwise_right_shifit)

_SP_DEFINE_EXPR_UNARY_OPERATOR(+, unary_plus)
_SP_DEFINE_EXPR_UNARY_OPERATOR(-, unary_minus)

_SP_DEFINE_EXPR_UNARY_OPERATOR(!, logical_not)
_SP_DEFINE_EXPR_BINARY_OPERATOR(&&, logical_and)
_SP_DEFINE_EXPR_BINARY_OPERATOR(||, logical_or)

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

#undef _SP_DEFINE_EXPR_BINARY_OPERATOR
#undef _SP_DEFINE_EXPR_UNARY_OPERATOR
#undef _SP_DEFINE_EXPR_BINARY_FUNCTION
#undef _SP_DEFINE_EXPR_UNARY_FUNCTION

#define _SP_DEFINE_COMPOUND_OP(_OP_)                              \
    template <typename TL, typename... TR>                        \
    TL &operator _OP_##=(TL &lhs, Expression<TR...> const &rhs) { \
        lhs = lhs _OP_ rhs;                                       \
        return lhs;                                               \
    }

_SP_DEFINE_COMPOUND_OP(+)
_SP_DEFINE_COMPOUND_OP(-)
_SP_DEFINE_COMPOUND_OP(*)
_SP_DEFINE_COMPOUND_OP(/)
_SP_DEFINE_COMPOUND_OP(%)
_SP_DEFINE_COMPOUND_OP(&)
_SP_DEFINE_COMPOUND_OP(|)
_SP_DEFINE_COMPOUND_OP (^)
_SP_DEFINE_COMPOUND_OP(<<)
_SP_DEFINE_COMPOUND_OP(>>)

#undef _SP_DEFINE_COMPOUND_OP

#define _SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(_OP_, _NAME_, _REDUCTION_)                                \
    namespace tags {                                                                                      \
    struct _NAME_ {                                                                                       \
        template <typename TL, typename TR>                                                               \
        static constexpr bool eval(TL const &l, TR const &r) {                                            \
            return ((l _OP_ r));                                                                          \
        }                                                                                                 \
        template <typename TL, typename TR>                                                               \
        constexpr bool operator()(TL const &l, TR const &r) const {                                       \
            return ((l _OP_ r));                                                                          \
        }                                                                                                 \
    };                                                                                                    \
    }                                                                                                     \
    template <typename... TL, typename TR>                                                                \
    bool operator _OP_(Expression<TL...> const &lhs, TR const &rhs) {                                     \
        return traits::reduction<_REDUCTION_>(Expression<tags::_NAME_, Expression<TL...>, TR>(lhs, rhs)); \
    };                                                                                                    \
    template <typename TL, typename... TR>                                                                \
    bool operator _OP_(TL const &lhs, Expression<TR...> const &rhs) {                                     \
        return traits::reduction<_REDUCTION_>(Expression<tags::_NAME_, TL, Expression<TR...>>(lhs, rhs)); \
    };                                                                                                    \
    template <typename... TL, typename... TR>                                                             \
    bool operator _OP_(Expression<TL...> const &lhs, Expression<TR...> const &rhs) {                      \
        return traits::reduction<_REDUCTION_>(                                                            \
            Expression<tags::_NAME_, Expression<TL...>, Expression<TR...>>(lhs, rhs));                    \
    };

_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(!=, not_equal_to, tags::logical_or)
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(==, equal_to, tags::logical_and)
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(<, less, tags::logical_and)
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(>, greater, tags::logical_and)
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(<=, less_equal, tags::logical_and)
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(>=, greater_equal, tags::logical_and)

//_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(real)                                          \
//_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(imag)                                          \
//
// template <typename T1>
// auto operator<<(T1 const &l, unsigned int r) {
//    return ((Expression<tags::shift_left, const T1, unsigned int>(l, r)));
//}
// template <typename T1>
// auto operator>>(T1 const &l, unsigned int r) {
//    return ((Expression<tags::shift_right, const T1, unsigned int>(l, r)));
//}

#undef _SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR
#undef _SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR
#undef _SP_DEFINE_EXPR_UNARY_BOOLEAN_OPERATOR

namespace tags {
struct _dot {};
struct _cross {};
}

template <typename TL, typename TR>
auto inner_product(TL const &l, TR const &r) {
    return traits::reduction<tags::addition>(l * r);
}
template <typename TL, typename TR>
auto dot_v(TL const &l, TR const &r) {
    return Expression<tags::_dot, const TL, const TR>(l, r);
}
template <typename TL, typename TR>
auto cross_v(TL const &l, TR const &r) {
    return Expression<tags::_cross, const TL, const TR>(l, r);
}

}  // namespace simpla

#endif  // SIMPLA_EXPRESSIONTEMPLATE_H
