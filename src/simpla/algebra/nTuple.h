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

#include "simpla/SIMPLA_config.h"

#include <cassert>

#include "ExpressionTemplate.h"
#include "simpla/utilities/type_traits.h"
#include "simpla/utilities/utility.h"
//#include "utility.h"
namespace simpla {
template <typename, int...>
struct nTuple;

template <typename...>
struct Expression;
}  // namespace simpla {
//
// namespace std {
//
// template <typename T, int... I>
// struct rank<simpla::nTuple<T, I...>> : public std::integral_constant<size_t, sizeof...(I)> {};
//
// template <typename V, int I0, int... I>
// struct extent<simpla::nTuple<V, I0, I...>> : public std::integral_constant<size_t, I0> {};
//
// template <typename...>
// struct extents;
// template <typename V, int... I>
// struct extents<simpla::nTuple<V, I...>> : public std::index_sequence<I...> {};
//}  // namespace std {

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
template <size_type I, typename U>
U const& get(U const& u) {
    return u;
}
template <size_type I, typename U>
U& get(U& u) {
    return u;
}
template <int N, typename T, int... M>
auto const& get(nTuple<T, M...> const& expr) {
    return expr[N];
}
template <size_type I, typename U, int... N>
auto& get(nTuple<U, N...>& u) {
    return u[I];
}
namespace detail {

template <typename TFun, typename TV, int N0>
__host__ __device__ void foreach_(nTuple<TV, N0>& m_data_, TFun const& fun) {
    for (int i = 0; i < N0; ++i) { fun(m_data_[i], i); }
}

template <typename TFun, typename TV, int N0, int N1>
__host__ __device__ void foreach_(nTuple<TV, N0, N1>& m_data_, TFun const& fun) {
    for (int i = 0; i < N0; ++i) {
        for (int j = 0; j < N1; ++j) { fun(m_data_[i][j], i, j); }
    }
}

template <typename TFun, typename TV, int N0, int N1, int N2>
__host__ __device__ void foreach_(nTuple<TV, N0, N1, N2>& m_data_, TFun const& fun) {
    for (int i = 0; i < N0; ++i) {
        for (int j = 0; j < N1; ++j) {
            for (int k = 0; k < N2; ++k) { fun(m_data_[i][j][k], i, j, k); }
        }
    }
}
}
template <typename TFun, typename TV, int... N>
__host__ __device__ void foreach (nTuple<TV, N...>& m_data_, TFun const& fun) {
    detail::foreach_(m_data_, fun);
}

template <int N0>
int recursive_calculate_shift(int s0) {
    return s0;
};
template <int N0, int... N, typename... Args>
int recursive_calculate_shift(int s0, Args&&... args) {
    return recursive_calculate_shift<N...>(std::forward<Args>(args)...) * N0 + s0;
};
template <typename TV, int... N>
struct extents<nTuple<TV, N...>> : public std::integral_constant<int, N...> {};

template <typename TV>
struct nt_size : public std::integral_constant<int, 1> {};
template <typename TV>
struct nt_size<nTuple<TV>> : public std::integral_constant<int, 1> {};
template <typename TV, int N0, int... N>
struct nt_size<nTuple<TV, N0, N...>> : public std::integral_constant<int, N0 * nt_size<nTuple<TV, N...>>::value> {};

template <int I, typename V>
V& nt_get_r(V& v) {
    return v;
};

template <int I, typename V, int N0, int... N>
decltype(auto) nt_get_r(nTuple<V, N0, N...>& v) {
    return nt_get_r<I / N0>(v[I % N0]);
};
template <int I, typename V>
V const& nt_get_r(V const& v) {
    return v;
};

template <int I, typename V, int N0, int... N>
decltype(auto) nt_get_r(nTuple<V, N0, N...> const& v) {
    return nt_get_r<I / N0>(v[I % N0]);
};
}  // namespace traits

///// n-dimensional primary type

// template <typename TV>
// struct nTuple<TV> {
//    typedef TV value_type;
//    typedef TV pod_type;
//};

template <typename TV, int N0, int... N>
struct nTuple<TV, N0, N...> {
    typedef nTuple<TV, N0, N...> this_type;

    typedef TV value_type;

    typedef typename std::conditional<sizeof...(N) == 0, TV, nTuple<TV, N...>>::type sub_type;

    sub_type m_data_[N0];

    __host__ __device__ nTuple() = default;
    __host__ __device__ ~nTuple() = default;
    template <typename U>
    __host__ __device__ nTuple(nTuple<U, N0, N...> const& other) {
        *this = other;
    };

    static constexpr size_type rank() { return sizeof...(N) + 1; }
    template <typename I>
    static constexpr size_type extents(I* d) {
        return traits::get_value(std::integer_sequence<int, N0, N...>(), d);
    }

    static constexpr size_type size() { return static_cast<size_type>(reduction_v(tags::multiplication(), N0, N...)); }

    __host__ __device__ nTuple(simpla::traits::nested_initializer_list_t<value_type, sizeof...(N) + 1> l) {
        simpla::traits::assign_nested_initializer_list<N0, N...>::apply(m_data_, l);
    }

    template <typename... U>
    __host__ __device__ nTuple(Expression<U...> const& expr) {
        for (int i = 0; i < N0; ++i) { m_data_[i] = traits::index(expr, i); }
    }

    __host__ __device__ nTuple(this_type const& other) {
        for (int i = 0; i < N0; ++i) { m_data_[i] = other.m_data_[i]; }
    }
    __host__ __device__ nTuple(this_type&& other) {
        for (int i = 0; i < N0; ++i) { m_data_[i] = other.m_data_[i]; }
    }
    __host__ __device__ void swap(this_type& other) {
        for (int i = 0; i < N0; ++i) { std::swap(m_data_[i], other.m_data_[i]); }
    }

    __host__ __device__ this_type& operator=(this_type const& rhs) {
        for (int i = 0; i < N0; ++i) { m_data_[i] = traits::index(rhs, i); }
        return (*this);
    }

    template <typename TR>
    __host__ __device__ this_type& operator=(TR const& rhs) {
        for (int i = 0; i < N0; ++i) { m_data_[i] = traits::index(rhs, i); }
        return (*this);
    }

    __host__ __device__ auto& operator[](int s) {
        assert(s < N0 && s >= 0);
        return m_data_[s];
    }

    __host__ __device__ auto const& operator[](int s) const {
        assert(s < N0 && s >= 0);
        return m_data_[s];
    }

    template <typename... Idx>
    __host__ __device__ auto const& at(Idx&&... idx) const {
        return traits::index(m_data_, std::forward<Idx>(idx)...);
    }

    template <typename... Idx>
    __host__ __device__ auto& at(Idx&&... idx) {
        return traits::index(m_data_, std::forward<Idx>(idx)...);
    }
};

template <>
template <typename TR>
__host__ __device__ nTuple<double, 3>& nTuple<double, 3>::operator=(TR const& rhs) {
#pragma clang loop unroll(full)
    for (int i = 0; i < 3; ++i) { m_data_[i] = traits::index(rhs, i); }
    return (*this);
}

template <>
template <typename TR>
__host__ __device__ nTuple<double, 9>& nTuple<double, 9>::operator=(TR const& rhs) {
#pragma clang loop unroll(full)
    for (int i = 0; i < 3; ++i) { m_data_[i] = traits::index(rhs, i); }
    return (*this);
}

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
    __host__ __device__ auto operator _OP_(nTuple<TL, NL...> const& lhs) {     \
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

template <typename TL, int... NL>
__host__ __device__ auto operator<<(nTuple<TL, NL...> const& lhs, unsigned int rhs) {
    return Expression<simpla::tags::bitwise_left_shift, nTuple<TL, NL...>, unsigned int>(lhs, rhs);
};
template <typename TL, int... NL>
__host__ __device__ auto operator>>(nTuple<TL, NL...> const& lhs, unsigned int rhs) {
    return Expression<simpla::tags::bitwise_right_shifit, nTuple<TL, NL...>, unsigned int>(lhs, rhs);
};
//_SP_DEFINE_NTUPLE_BINARY_OPERATOR(bitwise_left_shift, <<)
//_SP_DEFINE_NTUPLE_BINARY_OPERATOR(bitwise_right_shifit, >>)

_SP_DEFINE_NTUPLE_UNARY_OPERATOR(unary_plus, +)
_SP_DEFINE_NTUPLE_UNARY_OPERATOR(unary_minus, -)

_SP_DEFINE_NTUPLE_UNARY_OPERATOR(logical_not, !)
_SP_DEFINE_NTUPLE_BINARY_OPERATOR(logical_and, &&)
_SP_DEFINE_NTUPLE_BINARY_OPERATOR(logical_or, ||)
#undef _SP_DEFINE_NTUPLE_BINARY_OPERATOR
#undef _SP_DEFINE_NTUPLE_UNARY_OPERATOR

#define _SP_DEFINE_NTUPLE_BINARY_FUNCTION(_NAME_)                                                 \
    template <typename TL, int... NL, typename TR>                                                \
    __host__ __device__ auto _NAME_(nTuple<TL, NL...> const& lhs, TR const& rhs) {                \
        return Expression<tags::_NAME_, nTuple<TL, NL...>, TR>(lhs, rhs);                         \
    };                                                                                            \
    template <typename TL, typename TR, int... NR>                                                \
    __host__ __device__ auto _NAME_(TL const& lhs, nTuple<TR, NR...> const& rhs) {                \
        return Expression<tags::_NAME_, TL, nTuple<TR, NR...>>(lhs, rhs);                         \
    };                                                                                            \
    template <typename TL, int... NL, typename... TR>                                             \
    __host__ __device__ auto _NAME_(nTuple<TL, NL...> const& lhs, Expression<TR...> const& rhs) { \
        return Expression<tags::_NAME_, nTuple<TL, NL...>, Expression<TR...>>(lhs, rhs);          \
    };                                                                                            \
    template <typename... TL, typename TR, int... NR>                                             \
    __host__ __device__ auto _NAME_(Expression<TL...> const& lhs, nTuple<TR, NR...> const& rhs) { \
        return Expression<tags::_NAME_, Expression<TL...>, nTuple<TR, NR...>>(lhs, rhs);          \
    };                                                                                            \
    template <typename TL, int... NL, typename TR, int... NR>                                     \
    __host__ __device__ auto _NAME_(nTuple<TL, NL...> const& lhs, nTuple<TR, NR...> const& rhs) { \
        return Expression<tags::_NAME_, nTuple<TL, NL...>, nTuple<TR, NR...>>(lhs, rhs);          \
    };

#define _SP_DEFINE_NTUPLE_UNARY_FUNCTION(_NAME_)                  \
    template <typename T, int... N>                               \
    __host__ __device__ auto _NAME_(nTuple<T, N...> const& lhs) { \
        return Expression<tags::_NAME_, nTuple<T, N...>>(lhs);    \
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

#define _SP_DEFINE_NTUPLE_COMPOUND_OP(_OP_)                                                                         \
    template <typename TL, int... NL, typename TR>                                                                  \
    __host__ __device__ nTuple<TL, NL...>& operator _OP_##=(nTuple<TL, NL...>& lhs, TR const& rhs) {                \
        lhs = lhs _OP_ rhs;                                                                                         \
        return lhs;                                                                                                 \
    }                                                                                                               \
    template <typename TL, int... NL, typename... TR>                                                               \
    __host__ __device__ nTuple<TL, NL...>& operator _OP_##=(nTuple<TL, NL...>& lhs, Expression<TR...> const& rhs) { \
        lhs = lhs _OP_ rhs;                                                                                         \
        return lhs;                                                                                                 \
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

#define _SP_DEFINE_NTUPLE_BINARY_BOOLEAN_OPERATOR(_OP_, _NAME_, _REDUCTION_)                                \
    template <typename TL, int... NL, typename TR>                                                          \
    __host__ __device__ bool operator _OP_(nTuple<TL, NL...> const& lhs, TR const& rhs) {                   \
        return calculus::reduction<_REDUCTION_>(Expression<tags::_NAME_, nTuple<TL, NL...>, TR>(lhs, rhs)); \
    };                                                                                                      \
    template <typename TL, typename TR, int... NR>                                                          \
    __host__ __device__ bool operator _OP_(TL const& lhs, nTuple<TR, NR...> const& rhs) {                   \
        return calculus::reduction<_REDUCTION_>(Expression<tags::_NAME_, TL, nTuple<TR, NR...>>(lhs, rhs)); \
    };                                                                                                      \
    template <typename TL, int... NL, typename... TR>                                                       \
    __host__ __device__ bool operator _OP_(nTuple<TL, NL...> const& lhs, Expression<TR...> const& rhs) {    \
        return calculus::reduction<_REDUCTION_>(                                                            \
            Expression<tags::_NAME_, nTuple<TL, NL...>, Expression<TR...>>(lhs, rhs));                      \
    };                                                                                                      \
    template <typename... TL, typename TR, int... NR>                                                       \
    __host__ __device__ bool operator _OP_(Expression<TL...> const& lhs, nTuple<TR, NR...> const& rhs) {    \
        return calculus::reduction<_REDUCTION_>(                                                            \
            Expression<tags::_NAME_, Expression<TL...>, nTuple<TR, NR...>>(lhs, rhs));                      \
    };                                                                                                      \
    template <typename TL, int... NL, typename TR, int... NR>                                               \
    __host__ __device__ bool operator _OP_(nTuple<TL, NL...> const& lhs, nTuple<TR, NR...> const& rhs) {    \
        return calculus::reduction<_REDUCTION_>(                                                            \
            Expression<tags::_NAME_, nTuple<TL, NL...>, nTuple<TR, NR...>>(lhs, rhs));                      \
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
__host__ __device__ auto dot(TL const& l, TR const& r) {
    return calculus::reduction<tags::addition>(l * r);
}

template <typename T1, typename T2>
__host__ __device__ auto cross(T1 const& l, T2 const& r) {
    return traits::make_ntuple(traits::index(l, 1) * traits::index(r, 2) - traits::index(l, 2) * traits::index(r, 1),
                               traits::index(l, 2) * traits::index(r, 0) - traits::index(l, 0) * traits::index(r, 2),
                               traits::index(l, 0) * traits::index(r, 1) - traits::index(l, 1) * traits::index(r, 0));
}

}  // namespace simpla
#endif  // NTUPLE_H_
