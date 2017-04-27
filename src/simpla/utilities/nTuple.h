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
#include <simpla/algebra/Arithmetic.h>
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

template <typename T, int... I0>
struct reference<nTuple<T, I0...>> {
    typedef nTuple<T, I0...>& type;
};

template <typename T, int... I0>
struct reference<const nTuple<T, I0...>> {
    typedef nTuple<T, I0...> const& type;
};

template <typename T, int... I>
struct rank<nTuple<T, I...>> : public int_const<sizeof...(I)> {};

template <typename...>
struct extents;
template <typename V, int... I>
struct extents<nTuple<V, I...>> : public int_sequence<I...> {};

template <typename V, int I0, int... I>
struct extent<nTuple<V, I0, I...>> : public int_const<I0> {};

template <typename...>
struct value_type;

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

}  // namespace traits

namespace calculus {
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
    static void assign(nTuple<V, N0, N...>& lhs, TR& rhs) {
        for (int i = 0; i < N0; ++i) { getValue(lhs, i) = getValue(rhs, i); }
    };

    template <typename V, int N0, int... N>
    static void swap(nTuple<V, N0, N...>& lhs, nTuple<V, N0, N...>& rhs) {
        for (int i = 0; i < N0; ++i) { std::swap(getValue(lhs, i), getValue(rhs, i)); }
    };
};
}

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

    typename std::tuple<typename algebra::traits::reference<Args>::type...> m_args_;
    typedef std::true_type is_expression;
    typedef std::false_type prefer_pass_by_reference;
    typedef std::true_type prefer_pass_by_value;

    TOP m_op_;

    nTuple(this_type const& that) : m_args_(that.m_args_) {}

    nTuple(this_type&& that) noexcept : m_args_(that.m_args_) {}

    explicit nTuple(Args&... args) noexcept : m_args_(args...) {}

    virtual ~nTuple() = default;

    this_type& operator=(this_type const& that) = delete;
    this_type& operator=(this_type&& that) = delete;

    template <typename T>
    explicit operator T() const {
        return calculus::expr_parser<T, this_type>::eval(*this);
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

    typedef calculus::nTuple_calculator calculator;
    typedef std::true_type prefer_pass_by_reference;
    typedef TV value_type;

    typedef typename traits::sub_type_t<this_type> sub_type;

    sub_type data_[N0];

    nTuple() = default;
    ~nTuple() = default;

    explicit nTuple(simpla::traits::nested_initializer_list_t<value_type, sizeof...(NOthers) + 1> l) {
        simpla::traits::assign_nested_initializer_list<N0, NOthers...>::apply(data_, l);
    }

    template <typename... U>
    explicit nTuple(nTuple<_Expression<U...>> const& expr) {
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
    auto operator+(TR const& rhs) const {
        return nTuple<_Expression<algebra::tags::plus, const this_type, const TR>>(*this, rhs);
    };
    template <typename TR>
    auto operator-(TR const& rhs) const {
        return nTuple<_Expression<algebra::tags::minus, const this_type, const TR>>(*this, rhs);
    };
    template <typename TR>
    auto operator*(TR const& rhs) const {
        return nTuple<_Expression<algebra::tags::multiplies, const this_type, const TR>>(*this, rhs);
    };
    template <typename TR>
    auto operator/(TR const& rhs) const {
        return nTuple<_Expression<algebra::tags::divides, const this_type, const TR>>(*this, rhs);
    };

    auto operator-() const { return nTuple<_Expression<algebra::tags::negate, const this_type>>(*this); };
};

template <typename T, int N>
using Vector = nTuple<T, N>;

template <typename T, int M, int N>
using Matrix = nTuple<T, M, N>;

template <typename T, int... N>
using Tensor = nTuple<T, N...>;

}  // namespace simpla
#endif  // NTUPLE_H_
