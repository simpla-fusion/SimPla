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

#include "Algebra.h"
#include "Arithmetic.h"
#include "Expression.h"
#include <simpla/mpl/CheckConcept.h>

namespace simpla
{
namespace algebra { namespace declare { template<typename, size_type ...I> struct nTuple_; }}


/**
 * @ingroup algebra
 * @addtogroup ntuple n-tuple
 * @{
 *
 * @brief nTuple_ :n-tuple
 *
 * Semantics:
 *    n-tuple is a sequence (or ordered list) of n elements, where n is a positive
 *    integral. There is also one 0-tuple, an empty sequence. An n-tuple is defined
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
template<typename T, size_type ...N> using nTuple=algebra::declare::nTuple_<T, N...>;

template<typename T, size_type N> using Vector=algebra::declare::nTuple_<T, N>;

template<typename T, size_type M, size_type N> using Matrix=algebra::declare::nTuple_<T, M, N>;

template<typename T, size_type ...N> using Tensor=algebra::declare::nTuple_<T, N...>;


namespace algebra
{
namespace traits
{


template<typename T, size_type ...I>
struct rank<declare::nTuple_<T, I...> > : public index_const<sizeof...(I)> {};
template<typename V, size_type ...I>
struct extents<declare::nTuple_<V, I...> > : public index_sequence<I...> {};

template<typename T, size_type I0>
struct value_type<declare::nTuple_<T, I0> > { typedef T type; };

template<typename T, size_type ...I>
struct value_type<declare::nTuple_<T, I...> > { typedef T type; };

template<typename T> struct sub_type { typedef T type; };

template<typename T> using sub_type_t = typename sub_type<T>::type;

template<typename T, size_type I0, size_type  ...I>
struct sub_type<declare::nTuple_<T, I0, I...> >
{
    typedef std::conditional_t<sizeof...(I) == 0, T, declare::nTuple_<T, I...> > type;
};

template<typename ...> struct pod_type;

template<typename ...T> using pod_type_t = typename pod_type<T...>::type;

template<typename T> struct pod_type<T> { typedef T type; };

template<typename T, size_type I0>
struct pod_type<declare::nTuple_<T, I0> > { typedef T type[I0]; };

template<typename T, size_type I0, size_type  ...I>
struct pod_type<declare::nTuple_<T, I0, I...> > { typedef typename pod_type<declare::nTuple_<T, I...>>::type type[I0]; };

}//namespace traits

namespace declare { template<typename ...> struct Expression; }

template<typename ...T> struct _impl;

/// n-dimensional primary type
namespace declare
{


template<typename TV, size_type N0, size_type ...NOthers>
struct nTuple_<TV, N0, NOthers...>
{
public:
    typedef nTuple_<TV, N0, NOthers...> this_type;
    typedef traits::value_type_t<this_type> value_type;
    typedef traits::sub_type_t<this_type> sub_type;

    sub_type data_[N0];

    sub_type &operator[](size_type s) { return data_[s]; }

    sub_type const &operator[](size_type s) const { return data_[s]; }

    sub_type &at(size_type s) { return data_[s]; }

    sub_type const &at(size_type s) const { return data_[s]; }


public:

    template<typename TR>
    inline this_type &operator=(TR const &rhs)
    {
        _impl<this_type>::apply(tags::_assign(), (*this), rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator+=(TR const &rhs)
    {
        _impl<this_type>::apply(tags::plus_assign(), *this, rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator-=(TR const &rhs)
    {
        _impl<this_type>::apply(tags::minus_assign(), *this, rhs);

        return (*this);
    }

    template<typename TR>
    inline this_type &operator*=(TR const &rhs)
    {
        _impl<this_type>::apply(tags::multiplies_assign(), *this, rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator/=(TR const &rhs)
    {
        _impl<this_type>::apply(tags::divides_assign(), *this, rhs);
        return (*this);
    }

};
}//namespace declare
namespace _detail
{
template<typename TOP, typename TL, size_type I0, typename TR>
void _apply(TOP const &, declare::nTuple_<TL, I0> &lhs, TR const &rhs)
{
    typedef _impl<declare::nTuple_<TL, I0> > impl_type;
    for (size_type i = 0; i < I0; ++i)
    {
        TOP::eval(impl_type::get_value(lhs, i), impl_type::get_value(rhs, i));
    }
};

template<typename TOP, typename TL, size_type I0, size_type I1, typename TR>
void _apply(TOP const &, declare::nTuple_<TL, I0, I1> &lhs, TR const &rhs)
{
    typedef _impl<declare::nTuple_<TL, I0, I1> > impl_type;
    for (size_type i = 0; i < I0; ++i)
        for (size_type j = 0; j < I1; ++j)
        {
            TOP::eval(impl_type::get_value(lhs, i, j), impl_type::get_value(rhs, i, j));
        }
};

template<typename TOP, typename TL, size_type I0, size_type I1, size_type I2, typename TR>
void _apply(TOP const &, declare::nTuple_<TL, I0, I1, I2> &lhs, TR const &rhs)
{
    typedef _impl<declare::nTuple_<TL, I0, I1, I2> > impl_type;
    for (size_type i = 0; i < I0; ++i)
        for (size_type j = 0; j < I1; ++j)
            for (size_type k = 0; k < I2; ++k)
            {
                TOP::eval(impl_type::get_value(lhs, i, j, k), impl_type::get_value(rhs, i, j, k));
            }
};

template<typename TOP, typename TL, size_type I0, size_type I1, size_type I2, size_type I3, typename TR>
void _apply(TOP const &, declare::nTuple_<TL, I0, I1, I2, I3> &lhs, TR const &rhs)
{
    typedef _impl<declare::nTuple_<TL, I0, I1, I2, I3> > impl_type;
    for (size_type i = 0; i < I0; ++i)
        for (size_type j = 0; j < I1; ++j)
            for (size_type k = 0; k < I2; ++k)
                for (size_type l = 0; l < I3; ++l)
                {
                    TOP::eval(impl_type::get_value(lhs, i, j, k, l), impl_type::get_value(rhs, i, j, k, l));
                }
};
}//namespace _detail

template<typename V, size_type ...J>
struct _impl<declare::nTuple_<V, J...> >
{
//    template<typename U> static auto
//    get_value(U &lhs, size_type const *s)
//    DECL_RET_TYPE((get_value_(std::integral_constant<bool, simpla::traits::is_indexable<U, size_type>::value>(), lhs, s)));
//
//

    template<typename U, typename ...Others> static auto
    get_value(U &lhs, Others &&...others)
    DECL_RET_TYPE((algebra::get_v(lhs, std::forward<Others>(others)...)));

//    template<typename TOP, typename ...Others, size_type ... index, typename ...Idx> static auto
//    _invoke_helper(declare::Expression<TOP, Others...> const &expr, index_sequence<index...>, Idx &&... s)
//    DECL_RET_TYPE((TOP::eval(get_value(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)))
//
//    template<typename TOP, typename   ...Others, typename ...Idx> static auto
//    get_value(declare::Expression<TOP, Others...> const &expr, Idx &&... s)
//    DECL_RET_TYPE((_invoke_helper(expr, index_sequence_for<Others...>(), std::forward<Idx>(s)...)))
////
//
//    template<typename U, typename ...Idx> static U &
//    get_value_(std::integral_constant<bool, false> const &, U &lhs, Idx &&...idx) { return lhs; }
//
//    template<typename U> static V
//    get_value_(std::integral_constant<bool, false> const &, U const &lhs) { return static_cast<V>(lhs); }
//
//    template<typename U> static U &
//    get_value_(std::integral_constant<bool, true> const &, U &lhs) { return lhs; };
//
//    template<typename U> static auto
//    get_value_(std::integral_constant<bool, true> const &, U &lhs, size_type s)
//    -> decltype(lhs[s]) { return lhs[s]; };
//
//
//    template<typename U, typename ...Idx> static auto
//    get_value_(std::integral_constant<bool, true> const &, U &lhs, size_type s, Idx &&...idx)
//    DECL_RET_TYPE(get_value(lhs[s], std::forward<Idx>(idx)...))
//
//    template<typename U, typename ...Idx> static auto
//    get_value(U &lhs, Idx &&...idx)
//    DECL_RET_TYPE((get_value_(std::integral_constant<bool, simpla::traits::is_indexable<U, size_type>::value>(),
//                              lhs, std::forward<Idx>(idx)...)));
//
//
//    template<typename U> static U &
//    get_value_(std::integral_constant<bool, false> const &, U &lhs, size_type const *s) { return lhs; }
//
//
//    template<typename U> static auto
//    get_value_(std::integral_constant<bool, true> const &, U &lhs, size_type const *s)
//    -> decltype(get_value(lhs[*s], s + 1)) { return get_value(lhs[*s], s + 1); };
//

    template<typename TOP, typename TR>
    static void apply(TOP const &op, declare::nTuple_<V, J...> &lhs, TR const &rhs) { _detail::_apply(op, lhs, rhs); };
};

template<typename V, size_type ...I, typename ...Idx> V &
get_v(declare::nTuple_<V, I...> &v, Idx &&...s)
{
    return _impl<declare::nTuple_<V, I...> >::get_value(v, std::forward<Idx>(s)...);
};

template<typename V, size_type ...I, typename ...Idx> V const &
get_v(declare::nTuple_<V, I...> const &v, Idx &&...s)
{
    return _impl<declare::nTuple_<V, I...> >::get_value(v, std::forward<Idx>(s)...);
};


}//namespaace algebra
}//namespace simpla
#endif  // NTUPLE_H_
