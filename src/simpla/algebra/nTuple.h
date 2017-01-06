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

#include "Algebra.h"
#include "Arithmetic.h"
#include "Expression.h"

namespace simpla {
namespace algebra {

namespace declare {
template <typename, size_type...>
struct nTuple_;
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
template <typename T, size_type... I0>
struct reference<declare::nTuple_<T, I0...>> {
    typedef declare::nTuple_<T, I0...>& type;
};

template <typename T, size_type... I0>
struct reference<const declare::nTuple_<T, I0...>> {
    typedef declare::nTuple_<T, I0...> const& type;
};

template <typename T, size_type... I>
struct rank<declare::nTuple_<T, I...>> : public index_const<sizeof...(I)> {};

template <typename V, size_type... I>
struct extents<declare::nTuple_<V, I...>> : public index_sequence<I...> {};

template <typename V, size_type I0, size_type... I>
struct extent<declare::nTuple_<V, I0, I...>> : public index_const<I0> {};

template <typename T, size_type I0>
struct value_type<declare::nTuple_<T, I0>> {
    typedef T type;
};

template <typename T, size_type... I>
struct value_type<declare::nTuple_<T, I...>> {
    typedef T type;
};

template <typename T, size_type I0, size_type... I>
struct sub_type<declare::nTuple_<T, I0, I...>> {
    typedef std::conditional_t<sizeof...(I) == 0, T, declare::nTuple_<T, I...>> type;
};

template <typename T, size_type I0>
struct pod_type<declare::nTuple_<T, I0>> {
    typedef T type[I0];
};

template <typename T, size_type I0, size_type... I>
struct pod_type<declare::nTuple_<T, I0, I...>> {
    typedef typename pod_type<declare::nTuple_<T, I...>>::type type[I0];
};

template <typename...>
struct make_nTuple {
    typedef void type;
};

template <typename TV, size_type... I>
struct make_nTuple<TV, integer_sequence<size_type, I...>> {
    typedef declare::nTuple_<TV, I...> type;
};
template <typename TV>
struct make_nTuple<TV, integer_sequence<size_type>> {
    typedef TV type;
};

}  // namespace traits

/// n-dimensional primary type
namespace declare {
template <typename TV>
struct nTuple_<TV> {
    typedef TV value_type;
    typedef TV pod_type;
};

template <typename TV, size_type N0, size_type... NOthers>
struct nTuple_<TV, N0, NOthers...> {
    typedef nTuple_<TV, N0, NOthers...> this_type;

    typedef calculus::calculator<this_type> calculator;

    typedef traits::value_type_t<this_type> value_type;

    typedef simpla::traits::add_extents_t<TV, N0, NOthers...> pod_type;

    typedef simpla::traits::add_extents_t<TV, NOthers...> sub_type;

    pod_type data_;

    sub_type& operator[](size_type s) { return data_[s]; }

    sub_type const& operator[](size_type s) const { return data_[s]; }

    sub_type& at(size_type s) { return data_[s]; }

    sub_type const& at(size_type s) const { return data_[s]; }

    value_type& at(size_type const* s) { return calculator::get_value(data_, s); }

    value_type const& at(size_type const* s) const { return calculator::get_value(data_, s); }

    template <typename... Idx>
    value_type& at(Idx&&... s) {
        return calculator::get_value(data_, std::forward<Idx>(s)...);
    }

    template <typename... Idx>
    value_type const& at(Idx&&... s) const {
        return calculator::get_value(data_, std::forward<Idx>(s)...);
    }

    template <typename... Idx>
    value_type& operator()(Idx&&... s) {
        return calculator::get_value(data_, std::forward<Idx>(s)...);
    }

    template <typename... Idx>
    value_type const& operator()(Idx&&... s) const {
        return calculator::get_value(data_, std::forward<Idx>(s)...);
    }

    nTuple_() {}

    ~nTuple_() {}

    nTuple_(simpla::traits::nested_initializer_list_t<value_type, sizeof...(NOthers) + 1> l) {
        simpla::traits::assign_nested_initializer_list<N0, NOthers...>::apply(data_, l);
    }

    template <typename... U>
    nTuple_(Expression<U...> const& expr) {
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
namespace detail {
template <typename G, size_type...>
struct do_n_loops;

template <typename G, size_type I>
struct do_n_loops<G, I> {
    template <typename TOP, typename TL, typename TR>
    static void eval(TOP const& op, TL& lhs, TR& rhs) {
        for (size_type i = 0; i < I; ++i) { op(G::get_value(lhs, i), G::get_value(rhs, i)); }
    };
};

template <typename G, size_type I0, size_type I1>
struct do_n_loops<G, I0, I1> {
    template <typename TOP, typename TL, typename TR>
    static void eval(TOP const& op, TL& lhs, TR& rhs) {
        for (size_type i = 0; i < I0; ++i)
            for (size_type j = 0; j < I1; ++j) {
                op(G::get_value(lhs, i, j), G::get_value(rhs, i, j));
            }
    };
};

template <typename G, size_type I0, size_type I1, size_type I2>
struct do_n_loops<G, I0, I1, I2> {
    template <typename TOP, typename TL, typename TR>
    static void eval(TOP const& op, TL& lhs, TR& rhs) {
        for (size_type i = 0; i < I0; ++i)
            for (size_type j = 0; j < I1; ++j)
                for (size_type k = 0; k < I2; ++k) {
                    op(G::get_value(lhs, i, j, k), G::get_value(rhs, i, j, k));
                }
    };
};

template <typename G, size_type I0, size_type I1, size_type I2, size_type I3>
struct do_n_loops<G, I0, I1, I2, I3> {
    template <typename TOP, typename TL, typename TR>
    static void eval(TOP const& op, TL& lhs, TR& rhs) {
        for (size_type i = 0; i < I0; ++i)
            for (size_type j = 0; j < I1; ++j)
                for (size_type k = 0; k < I2; ++k)
                    for (size_type l = 0; l < I3; ++l) {
                        op(G::get_value(lhs, i, j, k, l), G::get_value(rhs, i, j, k, l));
                    }
    };
};
}

namespace st = simpla::traits;

template <typename V, size_type... J>
struct calculator<declare::nTuple_<V, J...>> {
    typedef declare::nTuple_<V, J...> self_type;

    typedef calculator<declare::nTuple_<V, J...>> this_type;

   public:
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

    template <typename T, size_type... N, typename Idx>
    static decltype(auto) get_value(declare::nTuple_<T, N...>& v, Idx const* idx) {
        return get_value(v.data_, idx);
    };

    template <typename T, size_type... N, typename Idx>
    static decltype(auto) get_value(declare::nTuple_<T, N...> const& v, Idx const* idx) {
        return get_value(v.data_, idx);
    };

    template <typename T, size_type... N, typename... Idx>
    static decltype(auto) get_value(declare::nTuple_<T, N...>& v, Idx&&... idx) {
        return get_value(v.data_, std::forward<Idx>(idx)...);
    };

    template <typename T, size_type... N, typename... Idx>
    static decltype(auto) get_value(declare::nTuple_<T, N...> const& v, Idx&&... idx) {
        return get_value(v.data_, std::forward<Idx>(idx)...);
    };

    template <typename TOP, typename... Others, size_type... index, typename... Idx>
    static decltype(auto) _invoke_helper(declare::Expression<TOP, Others...> const& expr,
                                         index_sequence<index...>, Idx&&... s) {
        return ((expr.m_op_(get_value(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)));
    }

    template <typename TOP, typename... Others, typename... Idx>
    static decltype(auto) get_value(declare::Expression<TOP, Others...> const& expr, Idx&&... s) {
        return ((_invoke_helper(expr, index_sequence_for<Others...>(), std::forward<Idx>(s)...)));
    }

    template <typename TOP, typename TR>
    static void apply(TOP const& op, self_type& lhs, TR& rhs) {
        detail::do_n_loops<this_type, J...>::eval(op, lhs, rhs);
    };
};

}  // namespace calculaus{
namespace traits {
template <typename...>
struct nTuple_traits;

template <typename T, size_type... N>
struct nTuple_traits<T, index_sequence<N...>> {
    typedef declare::nTuple_<T, N...> type;
    typedef calculus::calculator<type> calculator;
};
template <typename T>
struct nTuple_traits<T> : public nTuple_traits<value_type_t<T>, extents<T>> {};

}  // namespace traits{

//template <typename TL, typename TR>
//auto inner_product(TL const& lhs, TR const& rhs,
//                   ENABLE_IF((traits::is_nTuple<TL, TR>::value) &&
//                             (traits::rank<TL>::value == 1 && traits::rank<TR>::value == 1) &&
//                             (traits::extent<TL>::value == 3 || traits::extent<TR>::value == 3))) {
//    typedef traits::value_type_t<TL> value_type;
//    typedef declare::nTuple_<value_type, 3> type;
//    typedef calculus::calculator<type> calculator;
//
//    return calculator::get_value(lhs, 0) * calculator::get_value(rhs, 0) +
//           calculator::get_value(lhs, 1) * calculator::get_value(rhs, 1) +
//           calculator::get_value(lhs, 2) * calculator::get_value(rhs, 2);
//}

template <typename TL, typename TR>
auto inner_product(declare::nTuple_<TL, 3> const& lhs, declare::nTuple_<TR, 3> const& rhs) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

template <typename TL, typename TR>
auto cross(declare::nTuple_<TL, 3> const& lhs, declare::nTuple_<TR, 3> const& rhs) {
    return declare::nTuple_<decltype(lhs[0] * rhs[1]), 3>{
        lhs[1] * rhs[2] - lhs[2] * rhs[1], lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],

    };
}

}  // namespace algebra
}  // namespace simpla
#endif  // NTUPLE_H_
