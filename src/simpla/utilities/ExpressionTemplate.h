//
// Created by salmon on 17-4-28.
//

#ifndef SIMPLA_EXPRESSIONTEMPLATE_H
#define SIMPLA_EXPRESSIONTEMPLATE_H

#include <type_traits>
#include <utility>
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

namespace traits {

template <typename TExpr>
struct is_expression : public false_type {};

template <typename... T>
struct is_expression<Expression<T...>> : public true_type {};

template <typename TOP, typename... Args>
struct rank<Expression<TOP, Args...>> : public seq_max<int_sequence<rank<Args>::value...>> {};

template <int N, typename TOP, typename... Args>
struct extent<Expression<TOP, Args...>, N> : public seq_max<int_sequence<extent<Args, N>::value...>> {};

template <int N>
struct get_s {
    template <typename T>
    static T const &eval(T const &expr, ENABLE_IF((extent<T>::value == 0))) {
        return expr;
    }
    template <typename T>
    static auto const &eval(T const &expr, ENABLE_IF((extent<T>::value > 0))) {
        return expr[N];
    }

    template <typename TOP, typename... Args, int... index>
    static auto eval0_(Expression<TOP, Args...> const &expr, index_sequence<index...>) {
        return expr.m_op_(get_s<N>::eval(std::get<index>(expr.m_args_))...);
    }

    template <typename TOP, typename... Args>
    static auto eval(Expression<TOP, Args...> const &expr) {
        return eval0_(expr, index_sequence_for<Args...>());
    }
};

template <int N, typename T>
auto get(T const &expr) {
    return get_s<N>::eval(expr);
}
template <int N, typename T>
auto const &get(T const *expr) {
    return expr[N];
}
template <typename TReduction>
struct reduction_s {
    template <typename Arg0>
    static auto eval0_(Arg0 const &arg0) {
        return reduction_s<TReduction>::eval(arg0);
    };

    template <typename Arg0, typename Arg1, typename... Others>
    static auto eval0_(Arg0 const &arg0, Arg1 const &arg1, Others &&... others) {
        return TReduction::eval(eval0_(arg0), eval0_(arg1, std::forward<Others>(others)...));
    };

    template <typename TExpr>
    static auto const &eval1_(TExpr const &expr, int_sequence<>) {
        return expr;
    }

    template <typename TExpr, int... I>
    static auto eval1_(TExpr const &expr, int_sequence<I...>) {
        return eval0_(get<I>(expr)...);
    }
    template <typename TExpr>
    static auto eval(TExpr const &expr) {
        return eval1_(expr, make_int_sequence<extent<TExpr>::value>());
    }
};
template <typename TReduction, typename TExpr>
auto reduction(TExpr const &expr) {
    return reduction_s<TReduction>::eval(expr);
};

namespace _impl {

template <typename TL, typename TR>
void assign_(TL &lhs, TR const &rhs, int_sequence<>){};

template <typename TL, typename TR, int I0, int... I>
void assign_(TL &lhs, TR const &rhs, int_sequence<I0, I...>) {
    lhs[I0] = traits::get<I0>(rhs);
    assign_(lhs, rhs, int_sequence<I...>());
};
template <typename T>
void swap_(T &lhs, T &rhs, int_sequence<>){};

template <typename T>
void swap0_(T &lhs, T &rhs, ENABLE_IF((traits::rank<T>::value == 0))) {
    std::swap(lhs, rhs);
}

template <typename T>
void swap0_(T &lhs, T &rhs, ENABLE_IF((traits::rank<T>::value > 0))) {
    lhs.swap(rhs);
}

template <typename T, int I0, int... I>
void swap_(T &lhs, T &rhs, int_sequence<I0, I...>) {
    swap0_(lhs[I0], rhs[I0]);
    swap_(lhs, rhs, int_sequence<I...>());
};

}  // namespace _impl {

template <typename T>
void swap(T &lhs, T &rhs) {
    _impl::swap_(lhs, rhs, make_int_sequence<extent<T>::value>());
};
template <typename TL, typename TR>
void assign(TL &lhs, TR const &rhs) {
    _impl::assign_(lhs, rhs, make_int_sequence<extent<TL>::value>());
};
}  // namespace

template <typename TOP, typename... Args>
struct Expression<TOP, Args...> {
    typedef Expression<TOP, Args...> this_type;

    typename std::tuple<traits::reference_t<Args>...> m_args_;

    TOP m_op_;

    Expression(this_type const &that) : m_args_(that.m_args_) {}
    Expression(this_type &&that) noexcept : m_args_(that.m_args_) {}
    template <typename... U>
    explicit Expression(U &&... args) : m_args_(std::forward<U>(args)...) {}

    virtual ~Expression() = default;

    template <typename T>
    explicit operator T() const {
        return static_cast<T>(traits::reduction(*this));
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
