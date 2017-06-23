/**
 * Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id: nTuple_.h 990 2010-12-14 11:06:21Z salmon $
 * @file ntuple.h
 *
 *  created on: Jan 27, 2010
 *      Author: yuzhi
 */

#ifndef NTUPLE_H_
#define NTUPLE_H_

#include "ExpressionTemplate.h"
#include "type_traits.h"
#include "utility.h"
namespace simpla {
template <typename, int...>
struct nTuple;
}  // namespace simpla {

namespace std {

template <typename T, int... I>
struct rank<simpla::nTuple<T, I...>> : public std::integral_constant<size_t, sizeof...(I)> {};

template <typename V, int I0, int... I>
struct extent<simpla::nTuple<V, I0, I...>> : public std::integral_constant<size_t, I0> {};

template <typename...>
struct extents;
template <typename V, int... I>
struct extents<simpla::nTuple<V, I...>> : public std::index_sequence<I...> {};
}  // namespace std {

namespace simpla {

/**
 * @addtogroup ntuple n-tuple
 * @{
 *
 * @brief nTuple_ :n-tuple
 *
 * Semantics:
 *    n-tuple is a sequence (or ordered list) of n elements, where n is a
 *positive
 *    integral. There is also one 0-tuple, an empty sequence. An n-tuple is
 *defined
 *    inductively using the construction of an ordered pair. Tuples are usually
 *    written by listing the elements within parenthesis '( )' and separated by
 *    commas; for example, (2, 7, 4, 1, 7) denotes a 5-tuple.
 *      [ wiki http://en.wikipedia.org/wiki/Tuple]
 *
 * Implement:
 *
 * @code{
 *   template<typename T, int ... n> struct nTuple_;
 *
 *   nTuple_<double,5> t={1,2,3,4,5};
 *
 *   nTuple_<T,N...> primary ntuple
 *
 *   nTuple_<Expression<TOP,TExpr>> unary nTuple_ expression
 *
 *   nTuple_<Expression<TOP,TExpr1,TExpr2>> binary nTuple_ expression
 *
 *
 *
 *   nTuple_<T,N> equiv. build-in array T[N]
 *
 *   nTuple_<T,N,M> equiv. build-in array T[N][M]
 *
 *    @endcode}
 *
 **/
namespace traits {

template <typename T, int I0, int... I>
struct reference<nTuple<T, I0, I...>> {
    typedef nTuple<T, I0, I...> const& type;
};

template <typename T, int I0, int... I>
struct reference<const nTuple<T, I0, I...>> {
    typedef nTuple<T, I0, I...> const& type;
};

template <typename T, int I0>
struct value_type<nTuple<T, I0>> {
    typedef T type;
};

template <typename T, int... I>
struct value_type<nTuple<T, I...>> {
    typedef T type;
};

template <typename T>
struct sub_type {
    typedef T type;
};
template <typename T>
using sub_type_t = typename sub_type<T>::type;

template <typename T, int I0, int... I>
struct sub_type<nTuple<T, I0, I...>> {
    typedef std::conditional_t<sizeof...(I) == 0, T, nTuple<T, I...>> type;
};

template <typename...>
struct pod_type;
template <typename... T>
using pod_type_t = typename pod_type<T...>::type;
template <typename T>
struct pod_type<T> {
    typedef T type;
};

template <typename T, int I0>
struct pod_type<nTuple<T, I0>> {
    typedef T type[I0];
};

template <typename T, int I0, int... I>
struct pod_type<nTuple<T, I0, I...>> {
    typedef typename pod_type<nTuple<T, I...>>::type type[I0];
};

template <typename T0, typename... Others>
auto make_ntuple(T0 const& a0, Others&&... others) {
    return nTuple<T0, sizeof...(Others) + 1>{a0, others...};
};

template <int N, typename T, int... M>
auto const& get(nTuple<T, M...> const& expr) {
    return expr[N];
}
}  // namespace traits

struct nTuple_calculator {
    template <typename T>
    static T& getValue(T* v, int s) {
        return v[s];
    };

    template <typename T>
    static T const& getValue(T const* v, int s) {
        return v[s];
    };

    template <typename T, typename TI, typename... Idx>
    static auto getValue(T* v, TI s0, Idx&&... idx) {
        return getValue(v[s0], std::forward<Idx>(idx)...);
    };

    template <typename T>
    static T& getValue(T& v) {
        return v;
    };

    template <typename T, typename... Idx>
    static T& getValue(T& v, Idx&&... idx) {
        return v;
    };

    template <typename T, typename TI>
    static T getValue(T* v, TI const* s) {
        return getValue(v[*s], s + 1);
    };

    template <typename T, int N0, int... N, typename Idx>
    static T getValue(nTuple<T, N0, N...>& v, Idx const* idx) {
        return getValue(v.data_[idx[0]], idx + 1);
    };

    template <typename T, int N0, int... N, typename Idx>
    static auto getValue(nTuple<T, N0, N...> const& v, Idx const* idx) {
        return getValue(v.data_[idx[0]], idx + 1);
    };

    template <typename T, int N0, int... N, typename... Idx>
    static auto getValue(nTuple<T, N0, N...>& v, int s, Idx&&... idx) {
        return getValue(v.data_[s], std::forward<Idx>(idx)...);
    };

    template <typename T, int N0, int... N, typename... Idx>
    static auto getValue(nTuple<T, N0, N...> const& v, int s, Idx&&... idx) {
        return getValue(v.data_[s], std::forward<Idx>(idx)...);
    };
    //
    //    template <typename... T, typename... Idx>
    //    static auto getValue(Expression<tags::_nTuple_cross, T...> const& expr, int s, Idx&&... others) {
    //        return getValue(std::get<0>(expr.m_args_), (s + 1) % 3, std::forward<Idx>(others)...) *
    //                   getValue(std::get<1>(expr.m_args_), (s + 2) % 3, std::forward<Idx>(others)...) -
    //               getValue(std::get<0>(expr.m_args_), (s + 2) % 3, std::forward<Idx>(others)...) *
    //                   getValue(std::get<1>(expr.m_args_), (s + 1) % 3, std::forward<Idx>(others)...);
    //    }

    template <typename TOP, typename... Others, size_t... index, typename... Idx>
    static auto _invoke_helper(Expression<TOP, Others...> const& expr, std::index_sequence<index...>, Idx&&... s) {
        return ((expr.m_op_(getValue(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)));
    }

    template <typename TOP, typename... Others, typename... Idx>
    static auto getValue(Expression<TOP, Others...> const& expr, Idx&&... s) {
        return ((_invoke_helper(expr, std::index_sequence_for<Others...>(), std::forward<Idx>(s)...)));
    }
};

/// n-dimensional primary type

template <typename TV>
struct nTuple<TV> {
    typedef TV value_type;
    typedef TV pod_type;
};

template <typename TV, int N0, int... N>
struct nTuple<TV, N0, N...> {
    typedef nTuple<TV, N0, N...> this_type;

    typedef nTuple_calculator calculator;
    typedef TV value_type;

    typedef typename std::conditional<sizeof...(N) == 0, TV, nTuple<TV, N...>>::type sub_type;

    sub_type data_[N0];

    __host__ __device__ nTuple() = default;
    __host__ __device__ ~nTuple() = default;

    __host__ __device__ nTuple(simpla::traits::nested_initializer_list_t<value_type, sizeof...(N) + 1> l) {
        simpla::traits::assign_nested_initializer_list<N0, N...>::apply(data_, l);
    }

    template <typename... U>
    __host__ __device__ nTuple(Expression<U...> const& expr) {
        traits::assign(*this, expr);
    }

    //    nTuple_(this_type const &other) = delete;
    //    nTuple_(this_type &&other) = delete;

    __host__ __device__ sub_type& operator[](int s) { return data_[s]; }

    __host__ __device__ sub_type const& operator[](int s) const { return data_[s]; }

    __host__ __device__ sub_type& at(int s) { return data_[s]; }

    __host__ __device__ sub_type const& at(int s) const { return data_[s]; }

    __host__ __device__ value_type& at(int const* s) { return calculator::getValue(*this, s); }

    template <typename... Idx>
    __host__ __device__ auto at(Idx&&... s) {
        return calculator::getValue(*this, std::forward<Idx>(s)...);
    }

    template <typename... Idx>
    __host__ __device__ auto at(Idx&&... s) const {
        return calculator::getValue(*this, std::forward<Idx>(s)...);
    }

    template <typename... Idx>
    __host__ __device__ auto operator()(Idx&&... s) {
        return calculator::getValue(*this, std::forward<Idx>(s)...);
    }

    template <typename... Idx>
    __host__ __device__ auto operator()(Idx&&... s) const {
        return calculator::getValue(*this, std::forward<Idx>(s)...);
    }
    __host__ __device__ void swap(this_type& other) { traits::swap(*this, other); }

    __host__ __device__ this_type& operator=(this_type const& rhs) {
        traits::assign(*this, rhs);
        return (*this);
    }

    template <typename TR>
    __host__ __device__ this_type& operator=(TR const& rhs) {
        traits::assign(*this, rhs);
        return (*this);
    }
};

#define _SP_DEFINE_NTUPLE_BINARY_OPERATOR(_NAME_, _OP_)                                                  \
    template <typename TL, int... NL, typename TR>                                                       \
    __host__ __device__ auto operator _OP_(nTuple<TL, NL...> const& lhs, TR const& rhs) {                \
        return Expression<simpla::tags::_NAME_, nTuple<TL, NL...>, TR>(lhs, rhs);                        \
    };                                                                                                   \
    template <typename TL, typename TR, int... NR>                                                       \
    __host__ __device__ auto operator _OP_(TL const& lhs, nTuple<TR, NR...> const& rhs) {                \
        return Expression<simpla::tags::_NAME_, TL, nTuple<TR, NR...>>(lhs, rhs);                        \
    };                                                                                                   \
    template <typename TL, int... NL, typename... TR>                                                    \
    __host__ __device__ auto operator _OP_(nTuple<TL, NL...> const& lhs, Expression<TR...> const& rhs) { \
        return Expression<simpla::tags::_NAME_, nTuple<TL, NL...>, Expression<TR...>>(lhs, rhs);         \
    };                                                                                                   \
    template <typename... TL, typename TR, int... NR>                                                    \
    __host__ __device__ auto operator _OP_(Expression<TL...> const& lhs, nTuple<TR, NR...> const& rhs) { \
        return Expression<simpla::tags::_NAME_, Expression<TL...>, nTuple<TR, NR...>>(lhs, rhs);         \
    };                                                                                                   \
    template <typename TL, int... NL, typename TR, int... NR>                                            \
    __host__ __device__ auto operator _OP_(nTuple<TL, NL...> const& lhs, nTuple<TR, NR...> const& rhs) { \
        return Expression<simpla::tags::_NAME_, nTuple<TL, NL...>, nTuple<TR, NR...>>(lhs, rhs);         \
    };

#define _SP_DEFINE_NTUPLE_UNARY_OPERATOR(_NAME_, _OP_)                         \
    template <typename TL, int... NL>                                          \
    auto operator _OP_(nTuple<TL, NL...> const& lhs) {                         \
        return Expression<simpla::tags::_NAME_, const nTuple<TL, NL...>>(lhs); \
    };

_SP_DEFINE_NTUPLE_BINARY_OPERATOR(addition, +)
_SP_DEFINE_NTUPLE_BINARY_OPERATOR(subtraction, -)
_SP_DEFINE_NTUPLE_BINARY_OPERATOR(multiplication, *)
_SP_DEFINE_NTUPLE_BINARY_OPERATOR(division, /)
_SP_DEFINE_NTUPLE_BINARY_OPERATOR(modulo, %)

_SP_DEFINE_NTUPLE_UNARY_OPERATOR(bitwise_not, ~)
_SP_DEFINE_NTUPLE_BINARY_OPERATOR(bitwise_xor, ^)
_SP_DEFINE_NTUPLE_BINARY_OPERATOR(bitwise_and, &)
_SP_DEFINE_NTUPLE_BINARY_OPERATOR(bitwise_or, |)
_SP_DEFINE_NTUPLE_BINARY_OPERATOR(bitwise_left_shift, <<)
_SP_DEFINE_NTUPLE_BINARY_OPERATOR(bitwise_right_shifit, >>)

_SP_DEFINE_NTUPLE_UNARY_OPERATOR(unary_plus, +)
_SP_DEFINE_NTUPLE_UNARY_OPERATOR(unary_minus, -)

_SP_DEFINE_NTUPLE_UNARY_OPERATOR(logical_not, !)
_SP_DEFINE_NTUPLE_BINARY_OPERATOR(logical_and, &&)
_SP_DEFINE_NTUPLE_BINARY_OPERATOR(logical_or, ||)

#undef _SP_DEFINE_NTUPLE_BINARY_OPERATOR
#undef _SP_DEFINE_NTUPLE_UNARY_OPERATOR

#define _SP_DEFINE_NTUPLE_BINARY_FUNCTION(_NAME_)                                        \
    template <typename TL, int... NL, typename TR>                                       \
    auto _NAME_(nTuple<TL, NL...> const& lhs, TR const& rhs) {                           \
        return Expression<tags::_NAME_, nTuple<TL, NL...>, TR>(lhs, rhs);                \
    };                                                                                   \
    template <typename TL, typename TR, int... NR>                                       \
    auto _NAME_(TL const& lhs, nTuple<TR, NR...> const& rhs) {                           \
        return Expression<tags::_NAME_, TL, nTuple<TR, NR...>>(lhs, rhs);                \
    };                                                                                   \
    template <typename TL, int... NL, typename... TR>                                    \
    auto _NAME_(nTuple<TL, NL...> const& lhs, Expression<TR...> const& rhs) {            \
        return Expression<tags::_NAME_, nTuple<TL, NL...>, Expression<TR...>>(lhs, rhs); \
    };                                                                                   \
    template <typename... TL, typename TR, int... NR>                                    \
    auto _NAME_(Expression<TL...> const& lhs, nTuple<TR, NR...> const& rhs) {            \
        return Expression<tags::_NAME_, Expression<TL...>, nTuple<TR, NR...>>(lhs, rhs); \
    };                                                                                   \
    template <typename TL, int... NL, typename TR, int... NR>                            \
    auto _NAME_(nTuple<TL, NL...> const& lhs, nTuple<TR, NR...> const& rhs) {            \
        return Expression<tags::_NAME_, nTuple<TL, NL...>, nTuple<TR, NR...>>(lhs, rhs); \
    };

#define _SP_DEFINE_NTUPLE_UNARY_FUNCTION(_NAME_)               \
    template <typename T, int... N>                            \
    auto _NAME_(nTuple<T, N...> const& lhs) {                  \
        return Expression<tags::_NAME_, nTuple<T, N...>>(lhs); \
    }

_SP_DEFINE_NTUPLE_UNARY_FUNCTION(cos)
_SP_DEFINE_NTUPLE_UNARY_FUNCTION(acos)
_SP_DEFINE_NTUPLE_UNARY_FUNCTION(cosh)
_SP_DEFINE_NTUPLE_UNARY_FUNCTION(sin)
_SP_DEFINE_NTUPLE_UNARY_FUNCTION(asin)
_SP_DEFINE_NTUPLE_UNARY_FUNCTION(sinh)
_SP_DEFINE_NTUPLE_UNARY_FUNCTION(tan)
_SP_DEFINE_NTUPLE_UNARY_FUNCTION(tanh)
_SP_DEFINE_NTUPLE_UNARY_FUNCTION(atan)
_SP_DEFINE_NTUPLE_UNARY_FUNCTION(exp)
_SP_DEFINE_NTUPLE_UNARY_FUNCTION(log)
_SP_DEFINE_NTUPLE_UNARY_FUNCTION(log10)
_SP_DEFINE_NTUPLE_UNARY_FUNCTION(sqrt)
_SP_DEFINE_NTUPLE_BINARY_FUNCTION(atan2)
_SP_DEFINE_NTUPLE_BINARY_FUNCTION(pow)

#undef _SP_DEFINE_NTUPLE_BINARY_FUNCTION
#undef _SP_DEFINE_NTUPLE_UNARY_FUNCTION

#define _SP_DEFINE_NTUPLE_COMPOUND_OP(_OP_)                                                     \
    template <typename TL, int... NL, typename TR>                                              \
    nTuple<TL, NL...>& operator _OP_##=(nTuple<TL, NL...>& lhs, TR const& rhs) {                \
        lhs = lhs _OP_ rhs;                                                                     \
        return lhs;                                                                             \
    }                                                                                           \
    template <typename TL, int... NL, typename... TR>                                           \
    nTuple<TL, NL...>& operator _OP_##=(nTuple<TL, NL...>& lhs, Expression<TR...> const& rhs) { \
        lhs = lhs _OP_ rhs;                                                                     \
        return lhs;                                                                             \
    }

_SP_DEFINE_NTUPLE_COMPOUND_OP(+)
_SP_DEFINE_NTUPLE_COMPOUND_OP(-)
_SP_DEFINE_NTUPLE_COMPOUND_OP(*)
_SP_DEFINE_NTUPLE_COMPOUND_OP(/)
_SP_DEFINE_NTUPLE_COMPOUND_OP(%)
_SP_DEFINE_NTUPLE_COMPOUND_OP(&)
_SP_DEFINE_NTUPLE_COMPOUND_OP(|)
_SP_DEFINE_NTUPLE_COMPOUND_OP (^)
_SP_DEFINE_NTUPLE_COMPOUND_OP(<<)
_SP_DEFINE_NTUPLE_COMPOUND_OP(>>)

#undef _SP_DEFINE_NTUPLE_COMPOUND_OP

#define _SP_DEFINE_NTUPLE_BINARY_BOOLEAN_OPERATOR(_OP_, _NAME_, _REDUCTION_)                              \
    template <typename TL, int... NL, typename TR>                                                        \
    bool operator _OP_(nTuple<TL, NL...> const& lhs, TR const& rhs) {                                     \
        return traits::reduction<_REDUCTION_>(Expression<tags::_NAME_, nTuple<TL, NL...>, TR>(lhs, rhs)); \
    };                                                                                                    \
    template <typename TL, typename TR, int... NR>                                                        \
    bool operator _OP_(TL const& lhs, nTuple<TR, NR...> const& rhs) {                                     \
        return traits::reduction<_REDUCTION_>(Expression<tags::_NAME_, TL, nTuple<TR, NR...>>(lhs, rhs)); \
    };                                                                                                    \
    template <typename TL, int... NL, typename... TR>                                                     \
    bool operator _OP_(nTuple<TL, NL...> const& lhs, Expression<TR...> const& rhs) {                      \
        return traits::reduction<_REDUCTION_>(                                                            \
            Expression<tags::_NAME_, nTuple<TL, NL...>, Expression<TR...>>(lhs, rhs));                    \
    };                                                                                                    \
    template <typename... TL, typename TR, int... NR>                                                     \
    bool operator _OP_(Expression<TL...> const& lhs, nTuple<TR, NR...> const& rhs) {                      \
        return traits::reduction<_REDUCTION_>(                                                            \
            Expression<tags::_NAME_, Expression<TL...>, nTuple<TR, NR...>>(lhs, rhs));                    \
    };                                                                                                    \
    template <typename TL, int... NL, typename TR, int... NR>                                             \
    bool operator _OP_(nTuple<TL, NL...> const& lhs, nTuple<TR, NR...> const& rhs) {                      \
        return traits::reduction<_REDUCTION_>(                                                            \
            Expression<tags::_NAME_, nTuple<TL, NL...>, nTuple<TR, NR...>>(lhs, rhs));                    \
    };

_SP_DEFINE_NTUPLE_BINARY_BOOLEAN_OPERATOR(!=, not_equal_to, tags::logical_or)
_SP_DEFINE_NTUPLE_BINARY_BOOLEAN_OPERATOR(==, equal_to, tags::logical_and)
_SP_DEFINE_NTUPLE_BINARY_BOOLEAN_OPERATOR(<=, less_equal, tags::logical_and)
_SP_DEFINE_NTUPLE_BINARY_BOOLEAN_OPERATOR(>=, greater_equal, tags::logical_and)
_SP_DEFINE_NTUPLE_BINARY_BOOLEAN_OPERATOR(>, greater, tags::logical_and)
_SP_DEFINE_NTUPLE_BINARY_BOOLEAN_OPERATOR(<, less, tags::logical_and)
#undef _SP_DEFINE_NTUPLE_BINARY_BOOLEAN_OPERATOR

//    DEF_BOP(shift_left, <<)
//    DEF_BOP(shift_right, >>)

template <typename TL, typename TR>
auto dot(TL const& l, TR const& r) {
    return traits::reduction<tags::addition>(l * r);
}

template <typename T1, typename T2>
auto cross(T1 const& l, T2 const& r) {
    return traits::make_ntuple(nTuple_calculator::getValue(l, 1) * nTuple_calculator::getValue(r, 2) -
                                   nTuple_calculator::getValue(l, 2) * nTuple_calculator::getValue(r, 1),
                               nTuple_calculator::getValue(l, 2) * nTuple_calculator::getValue(r, 0) -
                                   nTuple_calculator::getValue(l, 0) * nTuple_calculator::getValue(r, 2),
                               nTuple_calculator::getValue(l, 0) * nTuple_calculator::getValue(r, 1) -
                                   nTuple_calculator::getValue(l, 1) * nTuple_calculator::getValue(r, 0));
}

// template <typename T>
// T vec_dot(nTuple<T, 3> const& l, nTuple<T, 3> const& r) {
//    return l[0] * r[0] + l[1] * r[1] + l[2] * r[2];
//}
//
// template <typename TL, int... NL, typename TR, int... NR>
// auto vec_dot(nTuple<TL, NL...> const& l, nTuple<TR, NR...> const& r) {
//    return abs(l * r);
//}
//
// template <typename T>
// T vec_dot(nTuple<T, 4> const& l, nTuple<T, 4> const& r) {
//    return l[0] * r[0] + l[1] * r[1] + l[2] * r[2] + l[3] * r[3];
//}
// template <typename T, int N>
// T vec_dot(nTuple<T, N> const& l, nTuple<T, N> const& r) {
//    T res = l[0] * r[0];
//    for (int i = 1; i < N; ++i) { res += l[i] * r[i]; }
//    return res;
//}

//
////
//// template<typename T, int N0> std::istream &
//// input(std::istream &is, nTuple <T, N0> &tv)
////{
////    for (int i = 0; i < N0 && is; ++i) { is >> tv[i]; }
////    return (is);
////}
////
//// template<typename T, int N0, int ...N> std::istream &
//// input(std::istream &is, nTuple<T, N0, N ...> &tv)
////{
////    for (int i = 0; i < N0 && is; ++i) { input(is, tv[i]); }
////    return (is);
////}
//
// namespace _detail {
// template <typename T, int... N>
// std::ostream& printNd_(std::ostream& os, T const& d, integer_sequence<int, N...> const&,
//                       ENABLE_IF((!concept::is_indexable<T>::value))) {
//    os << d;
//    return os;
//}
//
// template <typename T, int M, int... N>
// std::ostream& printNd_(std::ostream& os, T const& d, integer_sequence<int, M, N...> const&,
//                       ENABLE_IF((concept::is_indexable<T>::value))) {
//    os << "[";
//    printNd_(os, d[0], integer_sequence<int, N...>());
//    for (int i = 1; i < M; ++i) {
//        os << " , ";
//        printNd_(os, d[i], integer_sequence<int, N...>());
//    }
//    os << "]";
//
//    return os;
//}
//
// template <typename T>
// std::istream& input(std::istream& is, T& a) {
//    is >> a;
//    return is;
//}
//
// template <typename T, int M0, int... M>
// std::istream& input(std::istream& is, nTuple<T, M0, M...>& a) {
//    for (int n = 0; n < M0; ++n) { _detail::input(is, a[n]); }
//    return is;
//}
//
//}  // namespace _detail
//
// template <typename T, int... M>
// std::ostream& operator<<(std::ostream& os, nTuple<T, M...> const& v) {
//    return _detail::printNd_(os, v.data_, integer_sequence<int, M...>());
//}
//
// template <typename T, int... M>
// std::istream& operator>>(std::istream& is, nTuple<T, M...>& a) {
//    _detail::input(is, a);
//    return is;
//}
// template <typename T, int... M>
// std::ostream& operator<<(std::ostream& os, std::tuple<nTuple<T, M...>, nTuple<T, M...>> const& v) {
//    os << "{ " << std::get<0>(v) << " ," << std::get<1>(v) << "}";
//    return os;
//};

}  // namespace simpla
#endif  // NTUPLE_H_
