//
// Created by salmon on 17-4-28.
//

#ifndef SIMPLA_EXPRESSIONTEMPLATE_H
#define SIMPLA_EXPRESSIONTEMPLATE_H

#include <cmath>
#include <complex>
#include <tuple>
#include "host_define.h"
#include "type_traits.h"
#include "utility.h"

namespace simpla {
template <typename...>
class Expression;
template <typename TM, typename TV, int...>
class Field;
}

namespace std {
template <typename TOP, typename... Args>
struct rank<simpla::Expression<TOP, Args...>>
    : public simpla::traits::seq_max<std::index_sequence<rank<Args>::value...>> {};

template <size_t N, typename TOP, typename... Args>
struct extent<simpla::Expression<TOP, Args...>, N>
    : public simpla::traits::seq_max<std::index_sequence<extent<Args, N>::value...>> {};

//    template <typename TOP, typename... Args>
//    struct extent<simpla::Expression<TOP, Args...>>
//            : public std::integral_constant<
//                    int, simpla::traits::mt_min<int, std::extent<typename
//                    std::remove_cv<Args>::type>::value...>::value> {};
}

namespace simpla {

namespace traits {
template <typename TM, typename TV, int... I>
struct reference<Field<TM, TV, I...>> {
    typedef const Field<TM, TV, I...> &type;
};

template <typename TM, typename TV, int... I>
struct reference<const Field<TM, TV, I...>> {
    typedef const Field<TM, TV, I...> &type;
};

template <typename TExpr>
struct is_expression : public std::false_type {};

template <typename... T>
struct is_expression<Expression<T...>> : public std::true_type {};

}  // namespace

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
        return static_cast<T>(calculus::reduction(*this));
    }
};

namespace calculus {

template <typename TFun, typename TypeList, typename Enable = void>
struct _IndexHelper;

template <typename T, typename... Args>
decltype(auto) getValue(T &expr, Args &&... args) {
    return _IndexHelper<T, traits::type_list<std::remove_reference_t<Args>...>>::value(expr,
                                                                                       std::forward<Args>(args)...);
};

template <typename T>
struct _IndexHelper<T, traits::type_list<>> {
    static decltype(auto) value(T &v) { return v; };
};

template <typename T, typename _Arg0, typename... _Args>
struct _IndexHelper<
    T, traits::type_list<_Arg0, _Args...>,
    std::enable_if_t<std::is_arithmetic<T>::value || std::is_same<std::remove_cv_t<T>, std::complex<double>>::value ||
                     std::is_same<std::remove_cv_t<T>, std::complex<float>>::value>> {
    template <typename Arg0, typename... Args>
    static decltype(auto) value(T &v, Arg0 &&arg0, Args &&... args) {
        return v;
    };
};

template <typename T, typename _Arg0, typename... _Args>
struct _IndexHelper<T, traits::type_list<_Arg0, _Args...>,
                    std::enable_if_t<traits::is_invocable<T, _Arg0, _Args...>::value>> {
    template <typename Arg0, typename... Args>
    static decltype(auto) value(T &v, Arg0 &&arg0, Args &&... args) {
        return v(std::forward<Arg0>(arg0), std::forward<Args>(args)...);
    };
};

template <typename T, typename _Arg0, typename... _Args>
struct _IndexHelper<
    T, traits::type_list<_Arg0, _Args...>,
    std::enable_if_t<(!traits::is_invocable<T, _Arg0, _Args...>::value) && traits::is_indexable<T, _Arg0>::value>> {
    template <typename Arg0, typename... Args>
    static decltype(auto) value(T &v, Arg0 &&arg0, Args &&... args) {
        return getValue(v[arg0], std::forward<Args>(args)...);
    };
};

template <typename TOP, typename... Others, typename... _Args>
struct _IndexHelper<const Expression<TOP, Others...>, traits::type_list<_Args...>> {
    template <size_t... index, typename... Args>
    static decltype(auto) _invoke_helper(std::index_sequence<index...>, Expression<TOP, Others...> const &expr,
                                         Args &&... args) {
        return expr.m_op_(getValue(std::get<index>(expr.m_args_), std::forward<Args>(args)...)...);
    }
    template <size_t... index, typename... Args>
    static decltype(auto) value(Expression<TOP, Others...> const &expr, Args &&... args) {
        return _invoke_helper(std::index_sequence_for<Others...>(), expr, std::forward<Args>(args)...);
    };
};

// template <typename T, typename TI>
// static T getValue(T *v, TI const *s) {
//    return getValue(v[*s], s + 1);
//};
//
//    template <typename... T, typename... Idx>
//    static decltype(auto) getValue(Expression<tags::_nTuple_cross, T...> const& expr, int s, Idx&&... others) {
//        return getValue(std::get<0>(expr.m_args_), (s + 1) % 3, std::forward<Idx>(others)...) *
//                   getValue(std::get<1>(expr.m_args_), (s + 2) % 3, std::forward<Idx>(others)...) -
//               getValue(std::get<0>(expr.m_args_), (s + 2) % 3, std::forward<Idx>(others)...) *
//                   getValue(std::get<1>(expr.m_args_), (s + 1) % 3, std::forward<Idx>(others)...);
//    }

// template <typename TOP, typename... Others, typename... Idx>
// static decltype(auto) getValue(Expression<TOP, Others...> const &expr, Idx &&... s) {
//    return ((_invoke_helper(expr, std::index_sequence_for<Others...>(), std::forward<Idx>(s)...)));
//}
//
template <typename LHS, typename RHS, typename... Args>
void Assign(LHS &lhs, RHS const &rhs, Args &&... args) {
    getValue(lhs, std::forward<Args>(args)...) = getValue(rhs, std::forward<Args>(args)...);
};
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
struct _dot {};
struct _cross {};
}

template <typename TL, typename TR>
__host__ __device__ auto inner_product(TL const &l, TR const &r) {
    return calculus::reduction<tags::addition>(l * r);
}
template <typename TL, typename TR>
__host__ __device__ auto dot_v(TL const &l, TR const &r) {
    return Expression<tags::_dot, const TL, const TR>(l, r);
}
template <typename TL, typename TR>
__host__ __device__ auto cross_v(TL const &l, TR const &r) {
    return Expression<tags::_cross, const TL, const TR>(l, r);
}

}  // namespace simpla

#endif  // SIMPLA_EXPRESSIONTEMPLATE_H
