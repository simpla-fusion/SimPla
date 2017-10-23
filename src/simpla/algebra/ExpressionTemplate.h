//
// Created by salmon on 17-4-28.
//

#ifndef SIMPLA_EXPRESSIONTEMPLATE_H
#define SIMPLA_EXPRESSIONTEMPLATE_H

#include <cmath>
#include <complex>
#include <tuple>
#include "simpla/utilities/type_traits.h"
#include "simpla/utilities/utility.h"
namespace simpla {
template <typename...>
class Expression;
}

namespace std {

template <typename TOP, typename... Args>
struct rank<simpla::Expression<TOP, Args...>> : public std::integral_constant<int, simpla::max(rank<Args>::value...)> {
};

template <size_t N, typename TOP, typename... Args>
struct extent<simpla::Expression<TOP, Args...>, N>
    : public std::integral_constant<int, simpla::max(extent<Args>::value...)> {};
}

namespace simpla {
namespace traits {

template <typename TExpr>
struct is_expression : public std::false_type {};

template <typename... T>
struct is_expression<Expression<T...>> : public std::true_type {};

template <typename... U, typename I0>
struct IndexHelper_<const Expression<U...>, I0> {
    template <size_type... I>
    static decltype(auto) helper_(std::integer_sequence<size_t, I...>, Expression<U...> const &expr, I0 const &s) {
        return expr.m_op_(index(std::get<I>(expr.m_args_), s)...);
    };
    static decltype(auto) eval(Expression<U...> const &expr, I0 const &s) {
        return helper_(std::make_index_sequence<sizeof...(U) - 1>(), expr, s);
    }
};

template <typename... U, typename... Args>
struct InvokeHelper_<Expression<U...>, std::tuple<Args...>> {
    template <size_type... I, typename... Args2>
    static decltype(auto) helper_(std::integer_sequence<size_type, I...>, Expression<U...> const &expr,
                                  Args2 &&... args) {
        return expr.m_op_(invoke(std::get<I>(expr.m_args_), std::forward<Args2>(args)...)...);
    };

    template <typename... Args2>
    static decltype(auto) eval(Expression<U...> const &expr, Args2 &&... args) {
        return helper_(std::make_index_sequence<sizeof...(U) - 1>(), expr, std::forward<Args2>(args)...);
    }
};
}
namespace calculus {

template <int N>
struct get_s {
    template <typename T>
    __host__ __device__ static T const &eval(T const &expr, ENABLE_IF((std::extent<T>::value == 0))) {
        return expr;
    }
    template <typename T>
    __host__ __device__ static decltype(auto) eval(T const &expr, ENABLE_IF((std::extent<T>::value > 0))) {
        return expr[N];
    }

    template <typename TOP, typename... Args, size_t... index>
    __host__ __device__ static decltype(auto) eval0_(Expression<TOP, Args...> const &expr,
                                                     std::index_sequence<index...>) {
        return expr.m_op_(get_s<N>::eval(std::get<index>(expr.m_args_))...);
    }

    template <typename TOP, typename... Args>
    __host__ __device__ static decltype(auto) eval(Expression<TOP, Args...> const &expr) {
        return eval0_(expr, std::index_sequence_for<Args...>());
    }
};

template <int N, typename T>
__host__ __device__ auto get(T const &expr) {
    return get_s<N>::eval(expr);
}
template <int N, typename T>
__host__ __device__ auto const &get(T const *expr) {
    return expr[N];
}

template <typename TReduction>
struct reduction_s {
    template <typename Arg0>
    __device__ __host__ static decltype(auto) eval0_(Arg0 const &arg0) {
        return reduction_s<TReduction>::eval(arg0);
    };

    template <typename Arg0, typename Arg1, typename... Others>
    __device__ __host__ static decltype(auto) eval0_(Arg0 const &arg0, Arg1 const &arg1, Others &&... others) {
        return TReduction::eval(eval0_(arg0), eval0_(arg1, std::forward<Others>(others)...));
    };

    template <typename TExpr>
    __device__ __host__ static decltype(auto) eval1_(TExpr const &expr, std::index_sequence<>) {
        return expr;
    }

    template <typename TExpr, size_t... I>
    __device__ __host__ static decltype(auto) eval1_(TExpr const &expr, std::index_sequence<I...>) {
        return eval0_(get<I>(expr)...);
    }
    template <typename TExpr>
    __device__ __host__ static decltype(auto) eval(TExpr const &expr) {
        return eval1_(expr, std::make_index_sequence<std::extent<TExpr>::value>());
    }
};
template <typename TReduction, typename TExpr>
__device__ __host__ auto reduction(TExpr const &expr) {
    return reduction_s<TReduction>::eval(expr);
};
//template <typename TL, typename TR>
//struct contraction_s;
//
//template <typename T, int N, int M>
//struct contraction_s<nTuple<T, N>, nTuple<T, N, M>> {
//    typedef nTuple<T, N> lhs_type;
//    typedef nTuple<T, N, M> rhs_type;
//    typedef nTuple<T, M> res_type;
//    __device__ __host__ static auto eval(lhs_type const &lhs, rhs_type const &rhs) {
//        res_type res = lhs[0] * rhs[0];
//        for (int i = 1; i < N; ++i) { res += lhs[i] * rhs[i]; }
//        return res;
//    }
//};
//template <typename T, int N, int M>
//struct contraction_s<nTuple<T, N, M>, nTuple<T, M>> {
//    typedef nTuple<T, N> rhs_type;
//    typedef nTuple<T, N, M> lhs_type;
//    typedef nTuple<T, N> res_type;
//    __device__ __host__ static auto eval(lhs_type const &lhs, rhs_type const &rhs) {
//        res_type res;
//        for (int i = 0; i < N; ++i) { res[i] = reduction<tags::addition>(lhs[0] * rhs[0]); }
//        return res;
//    }
//};
//template <typename TL, typename TR>
//__device__ __host__ auto contraction(TL const &lhs, TR const &rhs) {
//    return contraction_s<TL, TR>::eval(lhs, rhs);
//};
};

template <typename TOP, typename... Args>
struct Expression<TOP, Args...> {
    typedef Expression<TOP, Args...> this_type;

    std::tuple<traits::reference_t<Args>...> m_args_;

    TOP m_op_;

    __host__ __device__ Expression(this_type const &that) : m_args_(that.m_args_) {}
    __host__ __device__ Expression(this_type &&that) noexcept : m_args_(that.m_args_) {}
    template <typename... U>
    __host__ __device__ explicit Expression(U &&... args) : m_args_(std::forward<U>(args)...) {}

    __host__ __device__ virtual ~Expression() = default;

    template <typename T>
    __host__ __device__ explicit operator T() const {
        return calculus::reduction(*this);
    }
};
namespace tags {
struct _assign {
    template <typename TL, typename TR>
    __host__ __device__ static void eval(TL &l, TR const &r) {
        l = r;
    }
    template <typename TL, typename TR>
    __host__ __device__ void operator()(TL &l, TR const &r) const {
        l = r;
    }
};
}

#define _SP_DEFINE_EXPR_BINARY_OPERATOR(_OP_, _NAME_)                                                    \
    namespace tags {                                                                                     \
    struct _NAME_ {                                                                                      \
        template <typename TL, typename TR>                                                              \
        __host__ __device__ static constexpr auto eval(TL const &l, TR const &r) {                       \
            return ((l _OP_ r));                                                                         \
        }                                                                                                \
        template <typename TL, typename TR>                                                              \
        __host__ __device__ constexpr auto operator()(TL const &l, TR const &r) const {                  \
            return ((l _OP_ r));                                                                         \
        }                                                                                                \
    };                                                                                                   \
    }                                                                                                    \
    template <typename... TL, typename TR>                                                               \
    __host__ __device__ auto operator _OP_(Expression<TL...> const &lhs, TR const &rhs) {                \
        return Expression<tags::_NAME_, Expression<TL...>, TR>(lhs, rhs);                                \
    };                                                                                                   \
    template <typename TL, typename... TR>                                                               \
    __host__ __device__ auto operator _OP_(TL const &lhs, Expression<TR...> const &rhs) {                \
        return Expression<tags::_NAME_, TL, Expression<TR...>>(lhs, rhs);                                \
    };                                                                                                   \
    template <typename... TL, typename... TR>                                                            \
    __host__ __device__ auto operator _OP_(Expression<TL...> const &lhs, Expression<TR...> const &rhs) { \
        return Expression<tags::_NAME_, Expression<TL...>, Expression<TR...>>(lhs, rhs);                 \
    };

#define _SP_DEFINE_EXPR_UNARY_OPERATOR(_OP_, _NAME_)                       \
    namespace tags {                                                       \
    struct _NAME_ {                                                        \
        template <typename TL>                                             \
        __host__ __device__ static constexpr auto eval(TL const &l) {      \
            return (_OP_(l));                                              \
        }                                                                  \
        template <typename TL>                                             \
        __host__ __device__ constexpr auto operator()(TL const &l) const { \
            return _OP_(l);                                                \
        }                                                                  \
    };                                                                     \
    }                                                                      \
    template <typename... T>                                               \
    __host__ __device__ auto operator _OP_(Expression<T...> const &lhs) {  \
        return Expression<tags::_NAME_, Expression<T...>>(lhs);            \
    }

#define _SP_DEFINE_EXPR_BINARY_FUNCTION(_NAME_)                                                   \
    namespace tags {                                                                              \
    struct _NAME_ {                                                                               \
        template <typename TL, typename TR>                                                       \
        __host__ __device__ static constexpr auto eval(TL const &l, TR const &r) {                \
            return (_NAME_(l, r));                                                                \
        }                                                                                         \
        template <typename TL, typename TR>                                                       \
        __host__ __device__ constexpr auto operator()(TL const &l, TR const &r) const {           \
            return (_NAME_(l, r));                                                                \
        }                                                                                         \
    };                                                                                            \
    }                                                                                             \
    template <typename... TL, typename TR>                                                        \
    __host__ __device__ auto _NAME_(Expression<TL...> const &lhs, TR const &rhs) {                \
        return Expression<tags::_NAME_, Expression<TL...>, TR>(lhs, rhs);                         \
    };                                                                                            \
    template <typename TL, typename... TR>                                                        \
    __host__ __device__ auto _NAME_(TL const &lhs, Expression<TR...> const &rhs) {                \
        return Expression<tags::_NAME_, TL, Expression<TR...>>(lhs, rhs);                         \
    };                                                                                            \
    template <typename... TL, typename... TR>                                                     \
    __host__ __device__ auto _NAME_(Expression<TL...> const &lhs, Expression<TR...> const &rhs) { \
        return Expression<tags::_NAME_, Expression<TL...>, Expression<TR...>>(lhs, rhs);          \
    };

#define _SP_DEFINE_EXPR_UNARY_FUNCTION(_NAME_)                             \
    namespace tags {                                                       \
    struct _NAME_ {                                                        \
        template <typename TL>                                             \
        __host__ __device__ static constexpr auto eval(TL const &l) {      \
            return (std::_NAME_(l));                                       \
        }                                                                  \
        template <typename TL>                                             \
        __host__ __device__ constexpr auto operator()(TL const &l) const { \
            return std::_NAME_(l);                                         \
        }                                                                  \
    };                                                                     \
    }                                                                      \
    template <typename... T>                                               \
    __host__ __device__ auto _NAME_(Expression<T...> const &lhs) {         \
        return Expression<tags::_NAME_, Expression<T...>>(lhs);            \
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

#define _SP_DEFINE_COMPOUND_OP(_OP_)                                                  \
    template <typename TL, typename... TR>                                            \
    __host__ __device__ TL &operator _OP_##=(TL &lhs, Expression<TR...> const &rhs) { \
        lhs = lhs _OP_ rhs;                                                           \
        return lhs;                                                                   \
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

#define _SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(_OP_, _NAME_, _REDUCTION_)                                  \
    namespace tags {                                                                                        \
    struct _NAME_ {                                                                                         \
        template <typename TL, typename TR>                                                                 \
        __host__ __device__ static constexpr bool eval(TL const &l, TR const &r) {                          \
            return ((l _OP_ r));                                                                            \
        }                                                                                                   \
        template <typename TL, typename TR>                                                                 \
        __host__ __device__ constexpr bool operator()(TL const &l, TR const &r) const {                     \
            return ((l _OP_ r));                                                                            \
        }                                                                                                   \
    };                                                                                                      \
    }                                                                                                       \
    template <typename... TL, typename TR>                                                                  \
    __host__ __device__ bool operator _OP_(Expression<TL...> const &lhs, TR const &rhs) {                   \
        return calculus::reduction<_REDUCTION_>(Expression<tags::_NAME_, Expression<TL...>, TR>(lhs, rhs)); \
    };                                                                                                      \
    template <typename TL, typename... TR>                                                                  \
    __host__ __device__ bool operator _OP_(TL const &lhs, Expression<TR...> const &rhs) {                   \
        return calculus::reduction<_REDUCTION_>(Expression<tags::_NAME_, TL, Expression<TR...>>(lhs, rhs)); \
    };                                                                                                      \
    template <typename... TL, typename... TR>                                                               \
    __host__ __device__ bool operator _OP_(Expression<TL...> const &lhs, Expression<TR...> const &rhs) {    \
        return calculus::reduction<_REDUCTION_>(                                                            \
            Expression<tags::_NAME_, Expression<TL...>, Expression<TR...>>(lhs, rhs));                      \
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
// auto operator<<(T1 const &l,   size_t r) {
//    return ((Expression<tags::shift_left, const T1,   size_t>(l, r)));
//}
// template <typename T1>
// auto operator>>(T1 const &l,   size_t r) {
//    return ((Expression<tags::shift_right, const T1, unsigned size_t>(l, r)));
//}

#undef _SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR
#undef _SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR
#undef _SP_DEFINE_EXPR_UNARY_BOOLEAN_OPERATOR

namespace tags {
struct dot {};
struct cross {};
}

template <typename TL, typename TR>
__host__ __device__ auto inner_product(TL const &l, TR const &r) {
    return calculus::reduction<tags::addition>(l * r);
}
template <typename TL, typename TR>
__host__ __device__ auto dot_v(TL const &l, TR const &r) {
    return Expression<tags::dot, const TL, const TR>(l, r);
}
template <typename TL, typename TR>
__host__ __device__ auto cross_v(TL const &l, TR const &r) {
    return Expression<tags::cross, const TL, const TR>(l, r);
}

auto reduction_v(tags::multiplication const &op) = delete;
auto reduction_v(tags::addition const &op) = delete;
auto reduction_v(tags::logical_or const &op) = delete;
auto reduction_v(tags::logical_and const &op) = delete;

template <typename TOP, typename Arg0>
auto reduction_v(TOP const &op, Arg0 &&arg0) {
    return arg0;
};
template <typename TOP, typename Arg0, typename Arg1>
auto reduction_v(TOP const &op, Arg0 &&arg0, Arg1 &&arg1) {
    return op(std::forward<Arg0>(arg0), std::forward<Arg1>(arg1));
};

template <typename TOP, typename Arg0, typename... Args>
auto reduction_v(TOP const &op, Arg0 &&arg0, Args &&... args) {
    return op(std::forward<Arg0>(arg0), reduction_v(op, std::forward<Args>(args)...));
};

}  // namespace simpla

#endif  // SIMPLA_EXPRESSIONTEMPLATE_H
