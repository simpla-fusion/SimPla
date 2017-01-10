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
#include <simpla/mpl/check_concept.h>
#include <simpla/mpl/integer_sequence.h>

#include "Algebra.h"
#include "Expression.h"

namespace simpla {
namespace algebra {

namespace declare {
template <typename, int...>
struct nTuple_;
template <typename...>
struct Expression;
}

/**
 * @ingroup algebra
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
 **/
namespace traits {

template <typename T, int... I0>
struct reference<declare::nTuple_<T, I0...>> {
    typedef declare::nTuple_<T, I0...>& type;
};

template <typename T, int... I0>
struct reference<const declare::nTuple_<T, I0...>> {
    typedef declare::nTuple_<T, I0...> const& type;
};

template <typename T, int... I>
struct rank<declare::nTuple_<T, I...>> : public index_const<sizeof...(I)> {};

template <typename V, int... I>
struct extents<declare::nTuple_<V, I...>> : public index_sequence<I...> {};

template <typename V, int I0, int... I>
struct extent<declare::nTuple_<V, I0, I...>> : public index_const<I0> {};

template <typename T, int I0>
struct value_type<declare::nTuple_<T, I0>> {
    typedef T type;
};

template <typename T, int... I>
struct value_type<declare::nTuple_<T, I...>> {
    typedef T type;
};


template <typename T>
struct sub_type {
    typedef T type;
};
template <typename T>
using sub_type_t = typename sub_type<T>::type;

template <typename T, int I0, int... I>
struct sub_type<declare::nTuple_<T, I0, I...>> {
    typedef std::conditional_t<sizeof...(I) == 0, T, declare::nTuple_<T, I...>> type;
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
struct pod_type<declare::nTuple_<T, I0>> {
    typedef T type[I0];
};

template <typename T, int I0, int... I>
struct pod_type<declare::nTuple_<T, I0, I...>> {
    typedef typename pod_type<declare::nTuple_<T, I...>>::type type[I0];
};

template <typename...>
struct make_nTuple {
    typedef void type;
};

template <typename TV, int... I>
struct make_nTuple<TV, simpla::integer_sequence<int, I...>> {
    typedef declare::nTuple_<TV, I...> type;
};
template <typename TV>
struct make_nTuple<TV, simpla::integer_sequence<int>> {
    typedef TV type;
};

}  // namespace traits

namespace calculus {

struct nTuple_calculator {
    template <typename T>
    static T& get_value(T* v, int s) {
        return v[s];
    };
    template <typename T>
    static T const& get_value(T const* v, int s) {
        return v[s];
    };

    template <typename T, typename TI, typename... Idx>
    static decltype(auto) get_value(T* v, TI s0, Idx&&... idx) {
        return get_value(v[s0], std::forward<Idx>(idx)...);
    };

    template <typename T>
    static decltype(auto) get_value(T& v) {
        return v;
    };

    template <typename T, typename... Idx>
    static decltype(auto) get_value(T& v, Idx&&... idx) {
        return v;
    };

    template <typename T, typename TI>
    static decltype(auto) get_value(T* v, TI const* s) {
        return get_value(v[*s], s + 1);
    };

    template <typename T, int... N, typename Idx>
    static decltype(auto) get_value(declare::nTuple_<T, N...>& v, Idx const* idx) {
        return get_value(v.data_[idx[0]], idx + 1);
    };

    template <typename T, int... N, typename Idx>
    static decltype(auto) get_value(declare::nTuple_<T, N...> const& v, Idx const* idx) {
        return get_value(v.data_[idx[0]], idx + 1);
    };

    template <typename T, int... N, typename... Idx>
    static decltype(auto) get_value(declare::nTuple_<T, N...>& v, int s, Idx&&... idx) {
        return get_value(v.data_[s], std::forward<Idx>(idx)...);
    };

    template <typename T, int... N, typename... Idx>
    static decltype(auto) get_value(declare::nTuple_<T, N...> const& v, int s, Idx&&...
    idx) {
        return get_value(v.data_[s], std::forward<Idx>(idx)...);
    };

    template <typename... T, typename... Idx>
    static decltype(auto) get_value(declare::Expression<tags::_nTuple_cross, T...> const& expr,
                                    int s, Idx&&... others) {
        return get_value(std::get<0>(expr.m_args_), (s + 1) % 3, std::forward<Idx>(others)...) *
                   get_value(std::get<1>(expr.m_args_), (s + 2) % 3, std::forward<Idx>(others)...) -
               get_value(std::get<0>(expr.m_args_), (s + 2) % 3, std::forward<Idx>(others)...) *
                   get_value(std::get<1>(expr.m_args_), (s + 1) % 3, std::forward<Idx>(others)...);
    }

    template <typename TOP, typename... Others, int... index, typename... Idx>
    static decltype(auto) _invoke_helper(declare::Expression<TOP, Others...> const& expr,
                                         index_sequence<index...>, Idx&&... s) {
        return ((expr.m_op_(get_value(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)));
    }

    template <typename TOP, typename... Others, typename... Idx>
    static decltype(auto) get_value(declare::Expression<TOP, Others...> const& expr, Idx&&... s) {
        return ((_invoke_helper(expr, index_sequence_for<Others...>(), std::forward<Idx>(s)...)));
    }

    template <typename V, int N0, int... N, typename TOP, typename TR>
    static void apply(TOP const& op, declare::nTuple_<V, N0, N...>& lhs, TR& rhs) {
        for (int i = 0; i < N0; ++i) { op(get_value(lhs, i), get_value(rhs, i)); }
    };
};
}

/// n-dimensional primary type
namespace declare {

template <typename TV>
struct nTuple_<TV> {
    typedef TV value_type;
    typedef TV pod_type;
};

template <typename TV, int N0, int... NOthers>
struct nTuple_<TV, N0, NOthers...> {
    typedef nTuple_<TV, N0, NOthers...> this_type;

    typedef calculus::nTuple_calculator calculator;

    typedef TV value_type;

    typedef typename traits::sub_type_t<this_type> sub_type;

    sub_type data_[N0];

    sub_type& operator[](int s) { return data_[s]; }

    sub_type const& operator[](int s) const { return data_[s]; }

    sub_type& at(int s) { return data_[s]; }

    sub_type const& at(int s) const { return data_[s]; }

    value_type& at(int const* s) { return calculator::get_value(*this, s); }

    value_type const& at(int const* s) const { return calculator::get_value(*this, s); }

    template <typename... Idx>
    decltype(auto) at(Idx&&... s) {
        return calculator::get_value(*this, std::forward<Idx>(s)...);
    }

    template <typename... Idx>
    decltype(auto) at(Idx&&... s) const {
        return calculator::get_value(*this, std::forward<Idx>(s)...);
    }

    template <typename... Idx>
    decltype(auto) operator()(Idx&&... s) {
        return calculator::get_value(*this, std::forward<Idx>(s)...);
    }

    template <typename... Idx>
    decltype(auto) operator()(Idx&&... s) const {
        return calculator::get_value(*this, std::forward<Idx>(s)...);
    }

    nTuple_() {}

    ~nTuple_() {}

    nTuple_(simpla::traits::nested_initializer_list_t<value_type, sizeof...(NOthers) + 1> l) {
        simpla::traits::assign_nested_initializer_list<N0, NOthers...>::apply(data_, l);
    }

    template <typename... U>
    explicit nTuple_(Expression<U...> const& expr) {
        calculator::apply(tags::_assign(), (*this), expr);
    }

    //    nTuple_(this_type const &other) = delete;
    //
    //    nTuple_(this_type &&other) = delete;

    void swap(this_type& other) { calculator::apply(tags::_swap(), (*this), other); }

    this_type& operator=(this_type const& rhs) {
        calculator::apply(tags::_assign(), (*this), rhs);
        return (*this);
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        calculator::apply(tags::_assign(), (*this), rhs);
        return (*this);
    }
};

}  // namespace declare

namespace calculus {

namespace st = simpla::traits;

template <typename TRes, typename TL, typename TR, typename TOP, typename TReduction>
TRes reduce(TRes init, TL const& lhs, TR const& rhs, TOP const& op, TReduction const& reduction,
            ENABLE_IF((std::max(traits::extent<TL>::value, traits::extent<TR>::value) > 1))) {
    static constexpr int N = std::max(traits::extent<TL>::value, traits::extent<TR>::value);
    TRes res = init;
    for (int i = 0; i < N; ++i) {
        res = reduction(res, reduce(init, nTuple_calculator::get_value(lhs, i),
                                    nTuple_calculator::get_value(rhs, i), op, reduction));
    }

    return res;
}
template <typename TRes, typename TL, typename TR, typename TOP, typename TReduction>
TRes reduce(TRes init, TL const& lhs, TR const& rhs, TOP const& op, TReduction const& reduction,
            ENABLE_IF((std::max(traits::extent<TL>::value, traits::extent<TR>::value) <= 1))) {
    return init;
}

template <typename V, int... N, typename TOP, typename... Args>
struct expr_parser<declare::nTuple_<V, N...>, declare::Expression<TOP, Args...>> {
    static declare::nTuple_<V, N...> eval(declare::Expression<TOP, Args...> const& expr) {
        declare::nTuple_<V, N...> res;
        res = expr;
        return std::move(res);
    };
};
template <typename TL, typename TR>
struct expr_parser<Real, declare::Expression<tags::_nTuple_dot, TL, TR>> {
    static Real eval(declare::Expression<tags::_nTuple_dot, TL, TR> const& expr) {
        static constexpr int N =
            std::max(traits::extent<TL>::value, traits::extent<TR>::value);
        Real res = 0.0;

        for (int i = 0; i < N; ++i) {
            res +=
                static_cast<Real>(dot(nTuple_calculator::get_value(std::get<0>(expr.m_args_), i),
                                      nTuple_calculator::get_value(std::get<1>(expr.m_args_), i)));
        }
        return res;
    };
};


}  // namespace calculaus{
}  // namespace algebra

template <typename T, int... N>
using nTuple = algebra::declare::nTuple_<T, N...>;

template <typename T, int N>
using Vector = algebra::declare::nTuple_<T, N>;

template <typename T, int M, int N>
using Matrix = algebra::declare::nTuple_<T, M, N>;

template <typename T, int... N>
using Tensor = algebra::declare::nTuple_<T, N...>;

}  // namespace simpla
#endif  // NTUPLE_H_
