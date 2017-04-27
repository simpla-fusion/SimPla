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

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/CheckConcept.h>
#include <simpla/utilities/integer_sequence.h>
#include <simpla/utilities/type_traits.h>

namespace simpla {
template <typename, int...>
struct nTuple;

template <typename...>
struct _Expression;

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
 *    TODO: nTuple should move to sub-directory utilities
 **/
namespace traits {
template <typename...>
struct reference;

template <typename T, int I0, int... I>
struct reference<nTuple<T, I0, I...>> {
    typedef nTuple<T, I0, I...>& type;
};

template <typename T, int I0, int... I>
struct reference<const nTuple<T, I0, I...>> {
    typedef nTuple<T, I0, I...> const& type;
};

template <typename T, int... I>
struct rank<nTuple<T, I...>> : public int_const<sizeof...(I)> {};

template <typename...>
struct extents;
template <typename V, int... I>
struct extents<nTuple<V, I...>> : public int_sequence<I...> {};

template <typename V, int I0, int... I>
struct extent<nTuple<V, I0, I...>> : public int_const<I0> {};

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

template <typename...>
struct make_nTuple {
    typedef void type;
};

template <typename TV, int... I>
struct make_nTuple<TV, simpla::integer_sequence<int, I...>> {
    typedef nTuple<TV, I...> type;
};
template <typename TV>
struct make_nTuple<TV, simpla::integer_sequence<int>> {
    typedef TV type;
};

template <typename T>
class is_scalar : public std::integral_constant<bool, std::is_arithmetic<T>::value> {};

template <typename T>
class is_nTuple : public std::integral_constant<bool, false> {};
template <typename T, int... N>
class is_nTuple<nTuple<T, N...>> : public std::integral_constant<bool, true> {};

}  // namespace traits

namespace tags {
class _nTuple_cross;
class _nTuple_dot;
}
template <typename TL, typename TR>
struct expr_parser;

template <typename TL, typename TR>
struct expr_parser<Real, nTuple<_Expression<tags::_nTuple_dot, TL, TR>>> {
    static Real eval(nTuple<_Expression<tags::_nTuple_dot, TL, TR>> const& expr) {
        static constexpr int N = std::max(traits::extent<TL>::value, traits::extent<TR>::value);
        Real res = 0.0;

        //        for (int i = 0; i < N; ++i) {
        //            res += static_cast<Real>(dot(nTuple_calculator::getValue(std::get<0>(expr.m_args_), i),
        //                                         nTuple_calculator::getValue(std::get<1>(expr.m_args_), i)));
        //        }
        return res;
    }
};

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
    static decltype(auto) getValue(T* v, TI s0, Idx&&... idx) {
        return getValue(v[s0], std::forward<Idx>(idx)...);
    };

    template <typename T>
    static decltype(auto) getValue(T& v) {
        return v;
    };

    template <typename T, typename... Idx>
    static decltype(auto) getValue(T& v, Idx&&... idx) {
        return v;
    };

    template <typename T, typename TI>
    static decltype(auto) getValue(T* v, TI const* s) {
        return getValue(v[*s], s + 1);
    };

    template <typename T, int N0, int... N, typename Idx>
    static decltype(auto) getValue(nTuple<T, N0, N...>& v, Idx const* idx) {
        return getValue(v.data_[idx[0]], idx + 1);
    };

    template <typename T, int N0, int... N, typename Idx>
    static decltype(auto) getValue(nTuple<T, N0, N...> const& v, Idx const* idx) {
        return getValue(v.data_[idx[0]], idx + 1);
    };

    template <typename T, int N0, int... N, typename... Idx>
    static decltype(auto) getValue(nTuple<T, N0, N...>& v, int s, Idx&&... idx) {
        return getValue(v.data_[s], std::forward<Idx>(idx)...);
    };

    template <typename T, int N0, int... N, typename... Idx>
    static decltype(auto) getValue(nTuple<T, N0, N...> const& v, int s, Idx&&... idx) {
        return getValue(v.data_[s], std::forward<Idx>(idx)...);
    };

    template <typename... T, typename... Idx>
    static decltype(auto) getValue(nTuple<_Expression<tags::_nTuple_cross, T...>> const& expr, int s, Idx&&... others) {
        return getValue(std::get<0>(expr.m_args_), (s + 1) % 3, std::forward<Idx>(others)...) *
                   getValue(std::get<1>(expr.m_args_), (s + 2) % 3, std::forward<Idx>(others)...) -
               getValue(std::get<0>(expr.m_args_), (s + 2) % 3, std::forward<Idx>(others)...) *
                   getValue(std::get<1>(expr.m_args_), (s + 1) % 3, std::forward<Idx>(others)...);
    }

    template <typename TOP, typename... Others, int... index, typename... Idx>
    static decltype(auto) _invoke_helper(nTuple<_Expression<TOP, Others...>> const& expr, int_sequence<index...>,
                                         Idx&&... s) {
        return ((expr.m_op_(getValue(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)));
    }

    template <typename TOP, typename... Others, typename... Idx>
    static decltype(auto) getValue(nTuple<_Expression<TOP, Others...>> const& expr, Idx&&... s) {
        return ((_invoke_helper(expr, int_sequence_for<Others...>(), std::forward<Idx>(s)...)));
    }

    template <typename V, int N0, int... N, typename TR>
    static void assign(nTuple<V, N0, N...>& lhs, TR& rhs){
        //        for (int i = 0; i < N0; ++i) { getValue(lhs, i) = getValue(rhs, i); }
    };

    template <typename V, int N0, int... N>
    static void swap(nTuple<V, N0, N...>& lhs, nTuple<V, N0, N...>& rhs) {
        for (int i = 0; i < N0; ++i) { std::swap(getValue(lhs, i), getValue(rhs, i)); }
    };
};

// namespace st = simpla::traits;
//
// template <typename TRes, typename TL, typename TR, typename TOP, typename TReduction>
// TRes reduce(TRes init, TL const& lhs, TR const& rhs, TOP const& op, TReduction const& reduction,
//            ENABLE_IF((std::max(traits::extent<TL>::value, traits::extent<TR>::value) > 1))) {
//    static constexpr int N = std::max(traits::extent<TL>::value, traits::extent<TR>::value);
//    TRes res = init;
//    for (int i = 0; i < N; ++i) {
//        res = reduction(
//            res, reduce(init, nTuple_calculator::getValue(lhs, i), nTuple_calculator::getValue(rhs, i), op,
//            reduction));
//    }
//
//    return res;
//}
//
// template <typename TRes, typename TL, typename TR, typename TOP, typename TReduction>
// TRes reduce(TRes init, TL const& lhs, TR const& rhs, TOP const& op, TReduction const& reduction,
//            ENABLE_IF((std::max(traits::extent<TL>::value, traits::extent<TR>::value) <= 1))) {
//    return init;
//}
// template <typename V, int... N, typename TOP, typename... Args>
// struct expr_parser<nTuple<V, N...>, nTupleExpression<TOP, Args...>> {
//    static nTuple<V, N...> eval(nTupleExpression<TOP, Args...> const& expr) {
//        nTuple<V, N...> res;
//        //        res = expr;
//        return std::move(res);
//    };
//};
//

/// n-dimensional primary type

template <typename TOP, typename... Args>
struct nTuple<_Expression<TOP, Args...>> {
    typedef nTuple<_Expression<TOP, Args...>> this_type;

    std::tuple<traits::reference_t<Args>...> m_args_;
    typedef std::true_type is_expression;
    typedef std::false_type prefer_pass_by_reference;
    typedef std::true_type prefer_pass_by_value;

    TOP m_op_;

    nTuple(Args const&... args) : m_args_((args)...) {}
    nTuple(this_type const& that) noexcept : m_args_(that.m_args_) {}
    nTuple(this_type&& that) noexcept : m_args_(that.m_args_) {}

    virtual ~nTuple() = default;

    this_type& operator=(this_type const& that) = delete;
    this_type& operator=(this_type&& that) = delete;

    template <typename T>
    explicit operator T() const {
        return expr_parser<T, this_type>::eval(*this);
    }
};
template <typename TV>
struct nTuple<TV> {
    typedef TV value_type;
    typedef TV pod_type;
};

template <typename TV, int N0, int... NOthers>
struct nTuple<TV, N0, NOthers...> {
    typedef nTuple<TV, N0, NOthers...> this_type;

    typedef nTuple_calculator calculator;
    typedef std::true_type prefer_pass_by_reference;
    typedef TV value_type;

    typedef typename traits::sub_type_t<this_type> sub_type;

    sub_type data_[N0];

    nTuple() = default;
    ~nTuple() = default;

    nTuple(simpla::traits::nested_initializer_list_t<value_type, sizeof...(NOthers) + 1> l) {
        simpla::traits::assign_nested_initializer_list<N0, NOthers...>::apply(data_, l);
    }

    template <typename... U>
    nTuple(nTuple<_Expression<U...>> const& expr) {
        calculator::assign((*this), expr);
    }

    //    nTuple_(this_type const &other) = delete;
    //    nTuple_(this_type &&other) = delete;

    sub_type& operator[](int s) { return data_[s]; }

    sub_type const& operator[](int s) const { return data_[s]; }

    sub_type& at(int s) { return data_[s]; }

    sub_type const& at(int s) const { return data_[s]; }

    value_type& at(int const* s) { return calculator::getValue(*this, s); }

    value_type const& at(int const* s) const { return calculator::getValue(*this, s); }

    template <typename... Idx>
    decltype(auto) at(Idx&&... s) {
        return calculator::getValue(*this, std::forward<Idx>(s)...);
    }

    template <typename... Idx>
    decltype(auto) at(Idx&&... s) const {
        return calculator::getValue(*this, std::forward<Idx>(s)...);
    }

    template <typename... Idx>
    decltype(auto) operator()(Idx&&... s) {
        return calculator::getValue(*this, std::forward<Idx>(s)...);
    }

    template <typename... Idx>
    decltype(auto) operator()(Idx&&... s) const {
        return calculator::getValue(*this, std::forward<Idx>(s)...);
    }

    void swap(this_type& other) { calculator::swap((*this), other); }

    this_type& operator=(this_type const& rhs) {
        calculator::assign((*this), rhs);
        return (*this);
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        calculator::assign((*this), rhs);
        return (*this);
    }
    template <typename TR>
    this_type& operator+=(TR const& rhs) {
        *this = *this + rhs;
        return (*this);
    }
    template <typename TR>
    this_type& operator-=(TR const& rhs) {
        *this = *this - rhs;
        return (*this);
    }
    template <typename TR>
    this_type& operator*=(TR const& rhs) {
        *this = *this * rhs;
        return (*this);
    }
    template <typename TR>
    this_type& operator/=(TR const& rhs) {
        *this = *this / rhs;
        return (*this);
    }

    template <typename TR>
    bool operator==(TR const& rhs) const {
        return false;
    }
};

#define DEF_BOP(_NAME_, _OP_)                                                                                        \
    namespace ntuple_tags {                                                                                          \
    struct _NAME_ {                                                                                                  \
        template <typename TL, typename TR>                                                                          \
        static constexpr auto eval(TL const& l, TR const& r) {                                                       \
            return ((l _OP_ r));                                                                                     \
        }                                                                                                            \
        template <typename TL, typename TR>                                                                          \
        constexpr auto operator()(TL const& l, TR const& r) const {                                                  \
            return ((l _OP_ r));                                                                                     \
        }                                                                                                            \
    };                                                                                                               \
    }                                                                                                                \
                                                                                                                     \
    template <typename TL, int... NL>                                                                                \
    auto operator _OP_(nTuple<TL, NL...> const& lhs, Real rhs) {                                                     \
        return nTuple<_Expression<ntuple_tags::_NAME_, const nTuple<TL, NL...>, Real>>(lhs, rhs);                    \
    };                                                                                                               \
    template <typename TR, int... NR>                                                                                \
    auto operator _OP_(Real lhs, nTuple<TR, NR...> const& rhs) {                                                     \
        return nTuple<_Expression<ntuple_tags::_NAME_, Real, const nTuple<TR, NR...>>>(lhs, rhs);                    \
    };                                                                                                               \
    template <typename TL, int... NL>                                                                                \
    auto operator _OP_(nTuple<TL, NL...> const& lhs, int rhs) {                                                      \
        return nTuple<_Expression<ntuple_tags::_NAME_, const nTuple<TL, NL...>, int>>(lhs, rhs);                     \
    };                                                                                                               \
    template <typename TR, int... NR>                                                                                \
    auto operator _OP_(int lhs, nTuple<TR, NR...> const& rhs) {                                                      \
        return nTuple<_Expression<ntuple_tags::_NAME_, int, const nTuple<TR, NR...>>>(lhs, rhs);                     \
    };                                                                                                               \
    template <typename TL, int... NL, typename TR, int... NR>                                                        \
    auto operator _OP_(nTuple<TL, NL...> const& lhs, nTuple<TR, NR...> const& rhs) {                                 \
        return nTuple<_Expression<ntuple_tags::_NAME_, const nTuple<TL, NL...>, const nTuple<TR, NR...>>>(lhs, rhs); \
    };

#define DEF_UOP(_NAME_, _OP_)                                                          \
    namespace ntuple_tags {                                                            \
    struct _NAME_ {                                                                    \
        template <typename TL>                                                         \
        static constexpr auto eval(TL const& l) {                                      \
            return ((_OP_ l));                                                         \
        }                                                                              \
        template <typename TL>                                                         \
        constexpr auto operator()(TL const& l) const {                                 \
            return ((_OP_ l));                                                         \
        }                                                                              \
    };                                                                                 \
    }                                                                                  \
                                                                                       \
    template <typename TL, int... NL>                                                  \
    auto operator _OP_(nTuple<TL, NL...> const& lhs) {                                 \
        return nTuple<_Expression<ntuple_tags::_NAME_, const nTuple<TL, NL...>>>(lhs); \
    };

DEF_BOP(plus, +)
DEF_BOP(minus, -)
DEF_BOP(multiplies, *)
DEF_BOP(divides, /)
DEF_BOP(modulus, %)
DEF_UOP(negate, -)
DEF_UOP(unary_plus, +)
DEF_BOP(bitwise_and, &)
DEF_BOP(bitwise_or, |)
DEF_BOP(bitwise_xor, ^)
DEF_UOP(bitwise_not, ~)
DEF_BOP(shift_left, <<)
DEF_BOP(shift_right, >>)
DEF_UOP(logical_not, !)
// DEF_BOP(logical_and, &&)
// DEF_BOP(logical_or, ||)
// DEF_BOP(not_equal_to, !=)
// DEF_BOP(greater, >)
// DEF_BOP(less, <)
// DEF_BOP(greater_equal, >=)
// DEF_BOP(less_equal, <=)
// DEF_BOP(equal_to, ==)

#undef DEF_UOP
#undef DEF_BOP

template <typename T, int N>
using Vector = nTuple<T, N>;

template <typename T, int M, int N>
using Matrix = nTuple<T, M, N>;

template <typename T, int... N>
using Tensor = nTuple<T, N...>;

typedef nTuple<Real, 3ul> point_type;  //!< DataType of configuration space point (coordinates i.e. (x,y,z) )

typedef nTuple<Real, 3ul> vector_type;

typedef std::tuple<point_type, point_type> box_type;  //! two corner of rectangle (or hexahedron ) , <lower ,upper>

typedef long difference_type;  //!< Data type of the difference between indices,i.e.  s = i - j

typedef nTuple<index_type, 3> index_tuple;
typedef nTuple<size_type, 3> size_tuple;

typedef std::tuple<index_tuple, index_tuple> index_box_type;

// typedef std::complex<Real> Complex;

typedef nTuple<Real, 3> Vec3;

typedef nTuple<Real, 3> CoVec3;

typedef nTuple<Integral, 3> IVec3;

typedef nTuple<Real, 3> RVec3;

// typedef nTuple<Complex, 3> CVec3;

template <typename T>
T vec_dot(nTuple<T, 3> const& l, nTuple<T, 3> const& r) {
    return l[0] * r[0] + l[1] * r[1] + l[2] * r[2];
}

template <typename TL, int... NL, typename TR, int... NR>
auto vec_dot(nTuple<TL, NL...> const& l, nTuple<TR, NR...> const& r) {
    return abs(l * r);
}

template <typename T>
T vec_dot(nTuple<T, 4> const& l, nTuple<T, 4> const& r) {
    return l[0] * r[0] + l[1] * r[1] + l[2] * r[2] + l[3] * r[3];
}
template <typename T, int N>
T vec_dot(nTuple<T, N> const& l, nTuple<T, N> const& r) {
    T res = l[0] * r[0];
    for (int i = 1; i < N; ++i) { res += l[i] * r[i]; }
    return res;
}

template <typename T>
T determinant(nTuple<T, 3> const& m) {
    return m[0] * m[1] * m[2];
}

template <typename T>
T determinant(nTuple<T, 4> const& m) {
    return m[0] * m[1] * m[2] * m[3];
}

template <typename T>
T determinant(nTuple<T, 3, 3> const& m) {
    return m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] * m[1][2] * m[2][0] -
           m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1] * m[0][2] - m[1][2] * m[2][1] * m[0][0];
}
template <typename TL, int... NL, typename TR, int... NR>
auto abs(nTuple<TL, NL...> const& l, nTuple<TR, NR...> const& r) {
    return std::sqrt(vec_dot(l, r));
}
template <typename T, int... N>
T abs(nTuple<T, N...> const& m) {
    return std::sqrt(vec_dot(m, m));
}

template <typename T>
T determinant(nTuple<T, 4, 4> const& m) {
    return m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1] * m[3][0] -
           m[0][3] * m[1][1] * m[2][2] * m[3][0] + m[0][1] * m[1][3] * m[2][2] * m[3][0] +
           m[0][2] * m[1][1] * m[2][3] * m[3][0] - m[0][1] * m[1][2] * m[2][3] * m[3][0] -
           m[0][3] * m[1][2] * m[2][0] * m[3][1] + m[0][2] * m[1][3] * m[2][0] * m[3][1] +
           m[0][3] * m[1][0] * m[2][2] * m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1] -
           m[0][2] * m[1][0] * m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1] +
           m[0][3] * m[1][1] * m[2][0] * m[3][2] - m[0][1] * m[1][3] * m[2][0] * m[3][2] -
           m[0][3] * m[1][0] * m[2][1] * m[3][2] + m[0][0] * m[1][3] * m[2][1] * m[3][2] +
           m[0][1] * m[1][0] * m[2][3] * m[3][2] - m[0][0] * m[1][1] * m[2][3] * m[3][2] -
           m[0][2] * m[1][1] * m[2][0] * m[3][3] + m[0][1] * m[1][2] * m[2][0] * m[3][3] +
           m[0][2] * m[1][0] * m[2][1] * m[3][3] - m[0][0] * m[1][2] * m[2][1] * m[3][3] -
           m[0][1] * m[1][0] * m[2][2] * m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3];
}

template <typename T1, typename T2>
//  nTuple<std::result_of_t<tags::multiplies::eval(traits::value_type_t < T1 > ,
//                                                                traits::value_type_t < T2 > )>, 3>
auto cross(T1 const& l, T2 const& r, ENABLE_IF(traits::is_nTuple<T1>::value&& traits::is_nTuple<T2>::value)) {
    return nTuple<std::result_of_t<ntuple_tags::multiplies::eval(traits::value_type_t<T1>, traits::value_type_t<T2>)>,
                  3>{traits::get_v(l, 1) * traits::get_v(r, 2) - traits::get_v(l, 2) * traits::get_v(r, 1),
                     traits::get_v(l, 2) * traits::get_v(r, 0) - traits::get_v(l, 0) * traits::get_v(r, 2),
                     traits::get_v(l, 0) * traits::get_v(r, 1) - traits::get_v(l, 1) * traits::get_v(r, 0)};
}
//
// template <typename T, int... N>
// auto mod(nTuple<T, N...> const &l) {
//    return std::sqrt(std::abs(inner_product(l, l)));
//}

template <typename TOP, typename T>
T reduce(T const& v, ENABLE_IF(traits::is_scalar<T>::value)) {
    return v;
}

template <typename TOP, typename T>
traits::value_type_t<T> reduce(T const& v, ENABLE_IF(traits::is_nTuple<T>::value)) {
    static constexpr int n = traits::extent<T>::value;

    traits::value_type_t<T> res = reduce<TOP>(traits::get_v(v, 0));

    for (int s = 1; s < n; ++s) { res = TOP::eval(res, reduce<TOP>(traits::get_v(v, s))); }

    return res;
}

template <typename TL, typename TR>
auto inner_product(TL const& l, TR const& r, ENABLE_IF(traits::is_nTuple<TL>::value&& traits::is_nTuple<TL>::value)) {
    return ((reduce<ntuple_tags::plus>(l * r)));
}

template <typename T>
auto normal(T const& l, ENABLE_IF(traits::is_nTuple<T>::value)) {
    return ((std::sqrt(inner_product(l, l))));
}

template <typename T>
auto abs(T const& l, ENABLE_IF(traits::is_nTuple<T>::value)) {
    return ((std::sqrt(inner_product(l, l))));
}

template <typename T>
auto NProduct(T const& v, ENABLE_IF(traits::is_nTuple<T>::value)) {
    return ((reduce<ntuple_tags::multiplies>(v)));
}

template <typename T>
auto NSum(T const& v, ENABLE_IF(traits::is_nTuple<T>::value)) {
    return ((reduce<ntuple_tags::plus>(v)));
}

//
// template<typename T, int N0> std::istream &
// input(std::istream &is, nTuple <T, N0> &tv)
//{
//    for (int i = 0; i < N0 && is; ++i) { is >> tv[i]; }
//    return (is);
//}
//
// template<typename T, int N0, int ...N> std::istream &
// input(std::istream &is, nTuple<T, N0, N ...> &tv)
//{
//    for (int i = 0; i < N0 && is; ++i) { input(is, tv[i]); }
//    return (is);
//}

namespace _detail {
template <typename T, int... N>
std::ostream& printNd_(std::ostream& os, T const& d, int_sequence<N...> const&,
                       ENABLE_IF((!simpla::concept::is_indexable<T>::value))) {
    os << d;
    return os;
}

template <typename T, int M, int... N>
std::ostream& printNd_(std::ostream& os, T const& d, int_sequence<M, N...> const&,
                       ENABLE_IF((simpla::concept::is_indexable<T>::value))) {
    os << "[";
    printNd_(os, d[0], int_sequence<N...>());
    for (int i = 1; i < M; ++i) {
        os << " , ";
        printNd_(os, d[i], int_sequence<N...>());
    }
    os << "]";

    return os;
}

template <typename T>
std::istream& input(std::istream& is, T& a) {
    is >> a;
    return is;
}

template <typename T, int M0, int... M>
std::istream& input(std::istream& is, nTuple<T, M0, M...>& a) {
    for (int n = 0; n < M0; ++n) { _detail::input(is, a[n]); }
    return is;
}

}  // namespace _detail

template <typename T, int... M>
std::ostream& operator<<(std::ostream& os, nTuple<T, M...> const& v) {
    return _detail::printNd_(os, v.data_, int_sequence<M...>());
}

template <typename T, int... M>
std::istream& operator>>(std::istream& is, nTuple<T, M...>& a) {
    _detail::input(is, a);
    return is;
}
template <typename T, int... M>
std::ostream& operator<<(std::ostream& os, std::tuple<nTuple<T, M...>, nTuple<T, M...>> const& v) {
    os << "{ " << std::get<0>(v) << " ," << std::get<1>(v) << "}";
    return os;
};

}  // namespace simpla
#endif  // NTUPLE_H_
