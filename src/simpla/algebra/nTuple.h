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
#include <simpla/mpl/CheckConcept.h>

#include "Algebra.h"
#include "Arithmetic.h"
#include "Expression.h"

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
template<typename T, size_type ...I0>
struct reference<declare::nTuple_<T, I0...> > { typedef declare::nTuple_<T, I0...> &type; };

template<typename T, size_type ...I0>
struct reference<const declare::nTuple_<T, I0...> > { typedef declare::nTuple_<T, I0...> const &type; };

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

template<typename ...T> struct algebra_parser;

/// n-dimensional primary type
namespace declare
{
template<typename TV>
struct nTuple_<TV>
{
    typedef TV value_type;
    typedef TV pod_type;
};

template<typename TV, size_type N0, size_type ...NOthers>
struct nTuple_<TV, N0, NOthers...>
{


    typedef nTuple_<TV, N0, NOthers...> this_type;

    typedef traits::value_type_t<this_type> value_type;

    typedef simpla::traits::add_extents_t<TV, N0, NOthers...> pod_type;

    typedef simpla::traits::add_extents_t<TV, NOthers...> sub_type;

    pod_type data_;

    inline sub_type &operator[](size_type s) { return data_[s]; }

    inline sub_type const &operator[](size_type s) const { return data_[s]; }

    inline sub_type &at(size_type s) { return data_[s]; }

    inline sub_type const &at(size_type s) const { return data_[s]; }

    nTuple_() {}

    ~nTuple_() {}

    nTuple_(simpla::traits::nested_initializer_list_t<value_type, sizeof...(NOthers) + 1> l)
    {
        simpla::traits::assign_nested_initializer_list<N0, NOthers...>::apply(data_, l);
    }

    template<typename ...U>
    nTuple_(Expression<U...> const &expr)
    {
        algebra_parser<this_type>::apply(tags::_assign(), (*this), expr);
    }

    nTuple_(this_type const &other) = delete;

    nTuple_(this_type &&other) = delete;

    void swap(this_type &other)
    {
        algebra_parser<this_type>::apply(tags::_swap(), (*this), other);
    }

    template<typename TR> inline this_type &
    operator=(TR const &rhs)
    {
        algebra_parser<this_type>::apply(tags::_assign(), (*this), rhs);
        return (*this);
    }

    template<typename TR> inline this_type &
    operator+=(TR const &rhs)
    {
        algebra_parser<this_type>::apply(tags::plus_assign(), *this, rhs);
        return (*this);
    }

    template<typename TR> inline this_type &
    operator-=(TR const &rhs)
    {
        algebra_parser<this_type>::apply(tags::minus_assign(), *this, rhs);

        return (*this);
    }

    template<typename TR> inline this_type &
    operator*=(TR const &rhs)
    {
        algebra_parser<this_type>::apply(tags::multiplies_assign(), *this, rhs);
        return (*this);
    }

    template<typename TR> inline this_type &
    operator/=(TR const &rhs)
    {
        algebra_parser<this_type>::apply(tags::divides_assign(), *this, rhs);
        return (*this);
    }

};
}//namespace declare
namespace _detail
{

template<typename TOP, typename TL, typename TR> inline void
_apply(TOP const &, declare::nTuple_<TL, 3> &lhs, TR &rhs)
{
    typedef algebra_parser<declare::nTuple_<TL, 3> > impl_type;
    TOP::eval(lhs[0], impl_type::get_value(rhs, 0));
    TOP::eval(lhs[1], impl_type::get_value(rhs, 1));
    TOP::eval(lhs[2], impl_type::get_value(rhs, 2));

};

template<typename TOP, typename TL, size_type I0, typename TR> inline void
_apply(TOP const &, declare::nTuple_<TL, I0> &lhs, TR &rhs)
{
    typedef algebra_parser<declare::nTuple_<TL, I0> > impl_type;
    for (size_type i = 0; i < I0; ++i)
    {
        TOP::eval(lhs[i], impl_type::get_value(rhs, i));
    }
};

template<typename TOP, typename TL, size_type I0, size_type I1, typename TR> inline void
_apply(TOP const &, declare::nTuple_<TL, I0, I1> &lhs, TR &rhs)
{
    typedef algebra_parser<declare::nTuple_<TL, I0, I1> > impl_type;
    for (size_type i = 0; i < I0; ++i)
        for (size_type j = 0; j < I1; ++j)
        {
            TOP::eval(impl_type::get_value(lhs, i, j), impl_type::get_value(rhs, i, j));
        }
};

template<typename TOP, typename TL, size_type I0, size_type I1, size_type I2, typename TR> inline void
_apply(TOP const &, declare::nTuple_<TL, I0, I1, I2> &lhs, TR &rhs)
{
    typedef algebra_parser<declare::nTuple_<TL, I0, I1, I2> > impl_type;
    for (size_type i = 0; i < I0; ++i)
        for (size_type j = 0; j < I1; ++j)
            for (size_type k = 0; k < I2; ++k)
            {
                TOP::eval(impl_type::get_value(lhs, i, j, k), impl_type::get_value(rhs, i, j, k));
            }
};

template<typename TOP, typename TL, size_type I0, size_type I1, size_type I2, size_type I3, typename TR> inline void
_apply(TOP const &, declare::nTuple_<TL, I0, I1, I2, I3> &lhs, TR &rhs)
{
    typedef algebra_parser<declare::nTuple_<TL, I0, I1, I2, I3>> impl_type;
    for (size_type i = 0; i < I0; ++i)
        for (size_type j = 0; j < I1; ++j)
            for (size_type k = 0; k < I2; ++k)
                for (size_type l = 0; l < I3; ++l)
                {
                    TOP::eval(impl_type::get_value(lhs, i, j, k, l), impl_type::get_value(rhs, i, j, k, l));
                }
};
}//namespace _detail
namespace st=simpla::traits;

template<typename V, size_type ...J>
struct algebra_parser<declare::nTuple_<V, J...> >
{


public:
    template<typename T> static constexpr inline T &
    get_value(T &v)
    {
        return v;
    };

    template<typename T, typename I0> static constexpr inline st::remove_all_extents_t<T, I0> &
    get_value(T &v, I0 const *s, ENABLE_IF((st::is_indexable<T, I0>::value)))
    {
        return get_value(v[*s], s + 1);
    };

    template<typename T, typename I0> static constexpr inline st::remove_all_extents_t<T, I0> &
    get_value(T &v, I0 const *s, ENABLE_IF((!st::is_indexable<T, I0>::value)))
    {
        return v;
    };
private:
    template<typename T, typename ...Args> static constexpr inline T &
    get_value_(std::integral_constant<bool, false> const &, T &v, Args &&...)
    {
        return v;
    }


    template<typename T, typename I0, typename ...Idx> static constexpr inline st::remove_extents_t<T, I0, Idx...> &
    get_value_(std::integral_constant<bool, true> const &, T &v, I0 const &s0, Idx &&...idx)
    {
        return get_value(v[s0], std::forward<Idx>(idx)...);
    };
public:
    template<typename T, typename I0, typename ...Idx> static constexpr inline st::remove_extents_t<T, I0, Idx...> &
    get_value(T &v, I0 const &s0, Idx &&...idx)
    {
        return get_value_(std::integral_constant<bool, st::is_indexable<T, I0>::value>(),
                          v, s0, std::forward<Idx>(idx)...);
    };

    template<typename T, size_type N> static constexpr inline T &
    get_value(declare::nTuple_<T, N> &v, size_type const &s0) { return v[s0]; };

    template<typename T, size_type N> static constexpr inline T const &
    get_value(declare::nTuple_<T, N> const &v, size_type const &s0) { return v[s0]; };
public:
    template<typename TOP, typename ...Others, size_type ... index, typename ...Idx> static constexpr inline auto
    _invoke_helper(declare::Expression<TOP, Others...> const &expr, index_sequence<index...>, Idx &&... s)
    DECL_RET_TYPE((TOP::eval(get_value(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)))

    template<typename TOP, typename   ...Others, typename ...Idx> static constexpr inline auto
    get_value(declare::Expression<TOP, Others...> const &expr, Idx &&... s)
    DECL_RET_TYPE((_invoke_helper(expr, index_sequence_for<Others...>(), std::forward<Idx>(s)...)))

    template<typename TOP, typename TR>
    static constexpr inline void apply(TOP const &op, declare::nTuple_<V, J...> &lhs, TR &rhs)
    {
        _detail::_apply(op, lhs, rhs);
    };
};


}//namespaace algebra
}//namespace simpla
#endif  // NTUPLE_H_
