/**
 * Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id: nTuple.h 990 2010-12-14 11:06:21Z salmon $
 * @file ntuple.h
 *
 *  created on: Jan 27, 2010
 *      Author: yuzhi
 */

#ifndef CORE_toolbox_NTUPLE_H_
#define CORE_toolbox_NTUPLE_H_

#include "Algebra.h"
#include "Arithmetic.h"


namespace simpla
{


/**
 * @ingroup calculus
 * @addtogroup ntuple n-tuple
 * @{
 *
 * @brief nTuple :n-tuple
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
 *   template<typename T, int ... n> struct nTuple;
 *
 *   nTuple<double,5> t={1,2,3,4,5};
 *
 *   nTuple<T,N...> primary ntuple
 *
 *   nTuple<Expression<TOP,TExpr>> unary nTuple expression
 *
 *   nTuple<Expression<TOP,TExpr1,TExpr2>> binary nTuple expression
 *
 *
 *
 *   nTuple<T,N> equiv. build-in array T[N]
 *
 *   nTuple<T,N,M> equiv. build-in array T[N][M]
 *
 *    @endcode}
 **/
/// n-dimensional primary type
template<typename, size_type ...I> struct nTuple;

template<typename TV, size_type N0, size_type ...NOthers>
struct nTuple<TV, N0, NOthers...>
{
private:

    typedef TV value_type;

    typedef std::conditional_t<sizeof...(NOthers) == 0, TV, nTuple<value_type, NOthers...> > sub_type;
    typedef nTuple<value_type, N0, NOthers...> this_type;


public:
    sub_type data_[N0];

    sub_type &operator[](size_type s) { return data_[s]; }

    sub_type const &operator[](size_type s) const { return data_[s]; }

    sub_type &at(size_type s) { return data_[s]; }

    sub_type const &at(size_type s) const { return data_[s]; }

private:
    template<typename TOP, typename TR>
    void assign_(TOP const *op, TR const &rhs)
    {
        for (int i = 0; i < N0; ++i) { data_[i] = (*op)(data_[i], _detail::get_value(rhs, i)); }
    }

    template<typename TR>
    void assign_(std::nullptr_t, TR const &rhs)
    {
        for (int i = 0; i < N0; ++i) { data_[i] = _detail::get_value(rhs, i); }
    }

public:

    template<typename TR>
    inline this_type &operator=(TR const &rhs)
    {

        assign_(nullptr, rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator+=(TR const &rhs)
    {
        assign_(reinterpret_cast<algebra::tags::plus *>(this), rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator-=(TR const &rhs)
    {
        assign_(reinterpret_cast<algebra::tags::minus *>(this), rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator*=(TR const &rhs)
    {
        assign_(reinterpret_cast<algebra::tags::multiplies *>(this), rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator/=(TR const &rhs)
    {
        assign_(reinterpret_cast<algebra::tags::divides *>(this), rhs);
        return (*this);
    }

    struct _detail;
};

template<typename TV, size_type N0, size_type ...NOthers>
struct nTuple<TV, N0, NOthers...>::_detail
{
    template<typename U> static U
    get_value(U const &u, size_type s, ENABLE_IF((std::is_convertible<U, TV>::value)))
    {
        return static_cast<TV>(u);
    }


private:

    template<typename TExpr, size_type ... index>
    static algebra::traits::value_type_t<TExpr>
    _invoke_helper(TExpr const &expr, index_sequence<index...>, size_type s)
    {
        return expr.m_op_(get_value(std::get<index>(expr.m_args_), s)...);
    }

    template<typename TExpr, size_type ... index>
    static algebra::traits::value_type_t<TExpr>
    _invoke_helper(TExpr &expr, index_sequence<index...>, size_type s)
    {
        return expr.m_op_(get_value(std::get<index>(expr.m_args_), s)...);
    }

public:
    template<typename TOP, typename   ...T>
    static algebra::traits::value_type_t<algebra::Expression<TOP, T...>>
    get_value(algebra::Expression<TOP, T...> const &expr, size_type s)
    {
        return _invoke_helper(expr, index_sequence_for<T...>(), s);
    }

    template<typename U, size_type J0, size_type ...J>
    static algebra::traits::value_type_t<nTuple<U, J0, J...> >
    get_value(nTuple<U, J0, J...> const &expr, size_type s0)
    {
        return expr[s0];
    }

    template<typename U, size_type J0, size_type ...J, typename ...Idx>
    static algebra::traits::value_type_t<nTuple<U, J0, J...> >
    get_value(nTuple<U, J0, J...> const &expr, size_type s0, Idx &&... s)
    {
        return get_value(expr[s0], std::forward<Idx>(s)...);
    }
};

template<typename T, size_type N> using Vector=nTuple<T, N>;

template<typename T, size_type M, size_type N> using Matrix=nTuple<nTuple<T, N>, M>;

}//namespace simpla{namespace algebra{
#endif  // CORE_toolbox_NTUPLE_H_
