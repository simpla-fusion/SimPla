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

#include <cstdbool>
#include <cstddef>
#include <type_traits>

#include "ExpressionTemplate.h"
#include "type_traits.h"
#include "mpl.h"

namespace simpla
{


/**
 * @ingroup toolbox
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
template<typename, size_type ...> struct nTuple;

template<typename ...> class Expression;

namespace traits
{
template<typename> struct primary_type;

template<typename> struct pod_type;

template<typename> struct is_ntuple { static constexpr bool value = false; };

template<typename T, size_type ...N> struct is_ntuple<nTuple<T, N...>> { static constexpr bool value = true; };

template<typename T> using is_ntuple_t=  std::enable_if_t<is_ntuple<T>::value>;


template<typename T, size_type M, size_type ...N>
struct reference<nTuple<T, M, N...>> { typedef nTuple<T, M, N...> const &type; };

template<typename T> struct reference<nTuple<T>> { typedef nTuple<T> type; };

template<typename  ...T, typename TI> auto index(nTuple<Expression<T...>> const &v, TI const &s) { return v[s]; }

}  // namespace traits




template<typename TV>
struct nTuple<TV>
{
    typedef TV value_type;

    typedef void sub_type;
};

template<typename TV, size_type N>
struct nTuple<TV, N>
{
private:

    typedef TV value_type;

    typedef nTuple<value_type, N> this_type;

    static constexpr size_type m_extent = N;

    typedef value_type sub_type;

public:
    value_type data_[N];

    auto &operator[](size_type s) { return data_[s]; }

    auto const &operator[](size_type s) const { return data_[s]; }


    auto &at(size_type s) { return data_[s]; }

    auto const &at(size_type s) const { return data_[s]; }

//    this_type &operator++()    {        ++data_[N - 1];        return *this;    }
//
//    this_type &operator--()    {        --data_[N - 1];        return *this;    }

//    template<typename U, size_type I> operator nTuple<U, I>() const
//    {
//        nTuple<U, I> res;
//        res = *this;
//        return std::move(res);
//    }

private:
    template<typename TOP, typename TR> void assign(TOP const &op, integer_sequence<TR> const &) {};

    template<typename OP, typename TR, TR I0, TR...I>
    void assign(OP const &op, integer_sequence<TR, I0, I...> const &)
    {
        op(data_[N - sizeof...(I) - 1], I0);
        assign(op, integer_sequence<TR, I...>());
    };


    template<typename Op, typename TR>
    void assign(Op const &op, TR const &rhs)
    {
        for (size_type s = 0; s < N; ++s) { op(data_[s], traits::index(rhs, s)); }
    }

public:

    template<typename TR> inline this_type &operator=(TR const &rhs)
    {
        assign(_impl::_assign(), rhs);
        return (*this);
    }

    template<typename TR> inline this_type &operator+=(TR const &rhs)
    {
        assign(_impl::plus_assign(), rhs);
        return (*this);
    }

    template<typename TR> inline this_type &operator-=(TR const &rhs)
    {
        assign(_impl::minus_assign(), rhs);
        return (*this);
    }

    template<typename TR> inline this_type &operator*=(TR const &rhs)
    {
        assign(_impl::multiplies_assign(), rhs);
        return (*this);
    }

    template<typename TR> inline this_type &operator/=(TR const &rhs)
    {
        assign(_impl::divides_assign(), rhs);
        return (*this);
    }


};

template<typename TV, size_type N, size_type ...M>
struct nTuple<TV, N, M...>
{
private:

    typedef TV value_type;

    typedef nTuple<value_type, N, M...> this_type;

    static constexpr size_type m_extent = N;

    typedef typename std::conditional<(sizeof...(M) == 0), value_type,
            nTuple<value_type, M...>>::type sub_type;
public:
    sub_type data_[m_extent];

    auto &operator[](size_type s) { return data_[s]; }

    auto const &operator[](size_type s) const { return data_[s]; }


    auto &at(size_type s) { return data_[s]; }

    auto const &at(size_type s) const { return data_[s]; }

//    this_type &operator++()    {        ++data_[N - 1];        return *this;    }
//
//    this_type &operator--()    {        --data_[N - 1];        return *this;    }

    template<typename U, size_type ...I> operator nTuple<U, I...>() const
    {
        nTuple<U, I...> res;
        res = *this;
        return std::move(res);
    }

private:
    template<typename TOP, typename TR> void assign(TOP const &op, integer_sequence<TR> const &) {};

    template<typename OP, typename TR, TR I0, TR...I>
    void assign(OP const &op, integer_sequence<TR, I0, I...> const &)
    {
        op(data_[N - sizeof...(I) - 1], I0);
        assign(op, integer_sequence<TR, I...>());
    };


    template<typename Op, typename TR>
    void assign(Op const &op, TR const &rhs)
    {
        for (size_type s = 0; s < N; ++s) { op(data_[s], traits::index(rhs, s)); }
    }

public:

    template<typename TR> inline this_type &operator=(TR const &rhs)
    {
        assign(_impl::_assign(), rhs);
        return (*this);
    }

    template<typename TR> inline this_type &operator+=(TR const &rhs)
    {
        assign(_impl::plus_assign(), rhs);
        return (*this);
    }

    template<typename TR> inline this_type &operator-=(TR const &rhs)
    {
        assign(_impl::minus_assign(), rhs);
        return (*this);
    }

    template<typename TR> inline this_type &operator*=(TR const &rhs)
    {
        assign(_impl::multiplies_assign(), rhs);
        return (*this);
    }

    template<typename TR> inline this_type &operator/=(TR const &rhs)
    {
        assign(_impl::divides_assign(), rhs);
        return (*this);
    }


};

template<typename ... T>
struct nTuple<Expression<T...>> : public Expression<T...>
{
    typedef nTuple<Expression<T...>> this_type;

    using Expression<T...>::m_op_;
    using Expression<T...>::args;
    using Expression<T...>::Expression;

    template<typename U, size_type ...N>
    operator nTuple<U, N...>() const
    {
        nTuple<U, N...> res;
        res = *this;
        return std::move(res);
    }


    template<typename ID>
    inline auto operator[](ID const &s) const { return at(s); }

private:
    template<typename ID, size_type ... index>
    auto _invoke_helper(ID s, index_sequence<index...>) const
    {
        return m_op_(traits::index(std::get<index>(args), s)...);
    }

public:
    template<typename ID> auto at(ID const &s) const
    {
        return _invoke_helper(s, typename make_index_sequence<sizeof...(T) - 1>::type());
    }


};


template<typename TOP, typename ARG0>
struct nTuple<Expression<TOP, ARG0>> : public Expression<TOP, ARG0>
{
    typedef nTuple<Expression<TOP, ARG0>> this_type;

    using Expression<TOP, ARG0>::m_op_;
    using Expression<TOP, ARG0>::args;
    using Expression<TOP, ARG0>::Expression;

    template<typename U, size_type ...N>
    operator nTuple<U, N...>() const
    {
        nTuple<U, N...> res;
        res = *this;
        return std::move(res);
    }


    template<typename ID>
    auto at(ID const &s) const { return m_op_(traits::index(std::get<0>(args), s)); }

    template<typename ID>
    inline auto operator[](ID const &s) const { return m_op_(traits::index(std::get<0>(args), s)); }


};

template<typename TOP, typename ARG0, typename ARG1>
struct nTuple<Expression<TOP, ARG0, ARG1>> : public Expression<TOP, ARG0, ARG1>
{
    typedef nTuple<Expression<TOP, ARG0, ARG1>> this_type;

    using Expression<TOP, ARG0, ARG1>::m_op_;
    using Expression<TOP, ARG0, ARG1>::args;
    using Expression<TOP, ARG0, ARG1>::Expression;

    template<typename U, size_type ...N>
    operator nTuple<U, N...>() const
    {
        nTuple<U, N...> res;
        res = *this;
        return std::move(res);
    }


    template<typename ID>
    auto at(ID const &s) const
    {
        return m_op_(traits::index(std::get<0>(args), s), traits::index(std::get<1>(args), s));
    }

    template<typename ID>
    inline auto operator[](ID const &s) const
    {
        return m_op_(traits::index(std::get<0>(args), s), traits::index(std::get<1>(args), s));
    }


};


namespace traits
{


template<typename T, size_type ...M, size_type N>
struct access<N, nTuple<T, M...> >
{

    static constexpr auto get(nTuple<T, M...> &v) { return v[N]; }

    static constexpr auto get(nTuple<T, M...> const &v) { return v[N]; }

    template<typename U> static void set(nTuple<T, M...> &v, U const &u) { get(v) = u; }

};

/**
 * C++11 <type_traits>
 * @ref http://en.cppreference.com/w/cpp/types/rank
 */

template<typename T, size_type ...N>
struct rank<nTuple<T, N...>> : public std::integral_constant<int, extents<nTuple<T, N...>>::size()> {};

template<typename TV, size_type ...M, size_type N>
struct extent<nTuple<TV, M...>, N> { typedef typename mpl::unpack_int_seq<N, int, M...>::type type; };

template<typename T, size_type ...N> struct key_type<nTuple<T, N...>> { typedef size_type type; };

template<typename T> struct extents;

template<typename TV, size_type ...M>
struct extents<nTuple<TV, M...> > : public index_sequence<M...> {};
// public seq_concat<, traits::extents<TV>>::type { };

namespace _impl
{

template<typename ...> struct make_pod_array;
template<typename ...> struct make_primary_nTuple;

template<typename TV>
struct make_pod_array<TV, index_sequence<>> { typedef TV type; };

template<typename TV, size_type N0, size_type ... N>
struct make_pod_array<TV, index_sequence<N0, N...>>
{
    typedef typename make_pod_array<TV, index_sequence<N...>>::type type[N0];
};

template<typename TV, size_type ... N>
struct make_primary_nTuple<TV, index_sequence<N...>> { typedef nTuple<TV, N...> type; };

template<typename TV> struct make_primary_nTuple<TV, index_sequence<>> { typedef TV type; };

template<typename ... T> using make_pod_array_t = typename make_pod_array<T...>::type;

template<typename ... T> using make_primary_nTuple_t = typename make_primary_nTuple<T...>::type;
}
// namespace _impl

template<typename T, size_type ...N>
struct primary_type<nTuple<T, N...>>
{
    typedef _impl::make_primary_nTuple_t<

            value_type_t<nTuple<T, N...>>,

            extents<nTuple<T, N...>>
    >
            type;

};




template<typename T> using ntuple_cast_t=typename primary_type<T>::type;

template<typename T, size_type ...N>
struct pod_type<nTuple<T, N...>> { typedef _impl::make_pod_array_t<T, index_sequence<N...>> type; };


namespace _impl
{
template<typename ...> struct extents_helper;

template<typename TOP> struct extents_helper<TOP> { typedef index_sequence<> type; };

template<typename TOP, typename First, typename ...Others>
struct extents_helper<TOP, First, Others...>
{
    typedef typename extents_helper<TOP, First,
            typename extents_helper<TOP, Others...>::type>::type type;
};
template<typename TOP, size_type ...N>
struct extents_helper<TOP, index_sequence<N...>, index_sequence<> >
{
    typedef index_sequence<N...> type;
};

template<typename TOP, size_type ...N>
struct extents_helper<TOP, index_sequence<>, index_sequence<N...> >
{
    typedef index_sequence<N...> type;
};

template<typename TOP>
struct extents_helper<TOP, index_sequence<>, index_sequence<> >
{
    typedef index_sequence<> type;
};
template<typename TOP, size_type ...N, size_type ...M>
struct extents_helper<TOP, index_sequence<N...>,
        index_sequence<M...> >
{
    static_assert(std::is_same<index_sequence<N...>,
            index_sequence<M...> >::value, "extent mismatch!");

    typedef index_sequence<N...> type;
};

}// namespace _impl


template<typename TOP, typename ... T>
struct extents<nTuple<Expression<TOP, T...> > >
        : public _impl::extents_helper<TOP, traits::extents<T>...>::type
{
    typedef typename _impl::extents_helper<TOP, traits::extents<T>...>::type type;
};

template<typename TV, size_type N> struct value_type<nTuple<TV, N> > { typedef traits::value_type_t<TV> type; };

template<typename TV, size_type ...M> struct value_type<nTuple<TV, M...> > { typedef traits::value_type_t<TV> type; };

template<typename TOP, typename ... T>
struct value_type<nTuple<Expression<TOP, T...> > >
{
    typedef traits::result_of_t<TOP(traits::value_type_t<T>...)> type;
};


template<typename ...T>
struct primary_type<nTuple<Expression<T...> >>
{
    typedef _impl::make_primary_nTuple_t<

            value_type_t<nTuple<Expression<T...> >>,

            extents<nTuple<Expression<T...>>>
    >
            type;

};

template<typename T, size_type N0, size_t...N> auto &index(nTuple<T, N0, N...> &v, size_type s) { return v[s]; }

template<typename T, size_type N0, size_t...N> auto &index(nTuple<T, N0, N...> const &v, size_type s) { return v[s]; }


namespace _impl
{
template<typename T, size_type ...M, typename ...Others> void
assigne_nTuple_helper(nTuple<T, M...> &lhs, std::integral_constant<int, 0> const, Others &&... others)
{
}


template<typename T, size_type N, size_type ...M, size_type I, typename T0, typename ...Others> void
assigne_nTuple_helper(nTuple<T, N, M...> &lhs, std::integral_constant<int, I> const, T0 const &a0,
                      Others &&... others)
{
    lhs[N - I] = a0;

    assigne_nTuple_helper(lhs, std::integral_constant<int, I - 1>(), std::forward<Others>(others)...);
}


}//namespace _impl{


template<typename T, size_type ...M, typename ...Others>
nTuple<T, 1 + sizeof...(Others), M...> make_nTuple(nTuple<T, M...> const &a0, Others &&... others)
{
    nTuple<T, 1 + sizeof...(Others), M...> res;

    _impl::assigne_nTuple_helper(res, std::integral_constant<int, 1 + sizeof...(Others)>(), a0,
                                 std::forward<Others>(others)...);

    return std::move(res);
}

template<typename T0, typename ...Others>
nTuple<T0, 1 + sizeof...(Others)> make_nTuple(T0 const &a0, Others &&... others)
{
    nTuple<T0, 1 + sizeof...(Others)> res;

    _impl::assigne_nTuple_helper(res, std::integral_constant<int, 1 + sizeof...(Others)>(), a0,
                                 std::forward<Others>(others)...);

    return std::move(res);
}

template<typename TInts, TInts ...N>
nTuple<TInts, sizeof...(N)> seq2ntuple(integer_sequence<TInts, N...>)
{
    return std::move(nTuple<TInts, sizeof...(N)>({N...}));
}

template<typename TV, size_type N, typename T1>
nTuple<TV, N> append_ntuple(T1 const &v0, TV const &v1)
{
    nTuple<TV, N> res;
    res = v0;
    res[N - 1] = v1;
    return std::move(res);
}

template<typename TV, size_type N, typename T2>
nTuple<TV, N + 1> join_ntuple(nTuple<TV, N> const &left, T2 right)
{
    nTuple<TV, N + 1> res;
    res = left;
    res[N] = right;
    return std::move(res);
}

template<typename T1, size_type N, typename T2, size_type M>
nTuple<T1, N + M> join_ntuple(nTuple<T1, N> const &left, nTuple<T2, M> right)
{
    nTuple<T1, N + M> res;
    for (size_type i = 0; i < N; ++i)
    {
        res[i] = left[i];
    }
    for (size_type i = 0; i < M; ++i)
    {
        res[i + N] = right[i];
    }
    return std::move(res);
}

}  // namespace traits

template<typename T, size_type N> using Vector=nTuple<T, N>;

template<typename T, size_type M, size_type N> using Matrix=nTuple<T, M, N>;

template<typename T, size_type ... N> using Tensor=nTuple<T, N...>;

template<typename T> inline auto determinant(nTuple<T, 3> const &m) { return (m[0] * m[1] * m[2]); }

template<typename T> inline auto determinant(nTuple<T, 4> const &m) { return (m[0] * m[1] * m[2] * m[3]); }

template<typename T> inline auto determinant(Matrix<T, 3, 3> const &m)
{
    return m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] * m[1][2] * m[2][0] -
           m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1] * m[0][2] - m[1][2] * m[2][1] * m[0][0];
}

template<typename T>
inline auto determinant(Matrix<T, 4, 4> const &m_t)
{
    auto &m = m_t.data_;
    return
            m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1] * m[3][0]//
            - m[0][3] * m[1][1] * m[2][2] * m[3][0] + m[0][1] * m[1][3]//
                                                      * m[2][2] * m[3][0] + m[0][2] * m[1][1] * m[2][3] * m[3][0] -
            m[0][1]//
            * m[1][2] * m[2][3] * m[3][0] - m[0][3] * m[1][2] * m[2][0] * m[3][1]//
            + m[0][2] * m[1][3] * m[2][0] * m[3][1] + m[0][3] * m[1][0] * m[2][2]//
                                                      * m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1] -
            m[0][2] * m[1][0]//
            * m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1] + m[0][3]//
                                                                          * m[1][1] * m[2][0] * m[3][2] -
            m[0][1] * m[1][3] * m[2][0] * m[3][2]//
            - m[0][3] * m[1][0] * m[2][1] * m[3][2] + m[0][0] * m[1][3] * m[2][1]//
                                                      * m[3][2] + m[0][1] * m[1][0] * m[2][3] * m[3][2] -
            m[0][0] * m[1][1]//
            * m[2][3] * m[3][2] - m[0][2] * m[1][1] * m[2][0] * m[3][3] + m[0][1]//
                                                                          * m[1][2] * m[2][0] * m[3][3] +
            m[0][2] * m[1][0] * m[2][1] * m[3][3]//
            - m[0][0] * m[1][2] * m[2][1] * m[3][3] - m[0][1] * m[1][0] * m[2][2]//
                                                      * m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3]//
            ;
}

template<typename T1, size_type ... N1, typename T2, size_type ... N2>
inline auto cross(nTuple<T1, N1...> const &l, nTuple<T2, N2...> const &r)
{
    nTuple<decltype(traits::index(l, 0) * traits::index(r, 0)), 3> res;

    res[0] = traits::index(l, 1) * traits::index(r, 2) - traits::index(l, 2) * traits::index(r, 1);
    res[1] = traits::index(l, 2) * traits::index(r, 0) - traits::index(l, 0) * traits::index(r, 2);
    res[2] = traits::index(l, 0) * traits::index(r, 1) - traits::index(l, 1) * traits::index(r, 0);

    return std::move(res);

}


namespace _impl
{

template<int...> struct value_in_range;

template<size_type N, size_type ...DIMS>
struct value_in_range<N, DIMS...>
{
    template<typename T0, typename T1, typename T2>
    static bool eval(T0 const &b, T1 const &e, T2 const &x)
    {

        for (size_type i = 0; i < N; ++i) { if (!value_in_range<DIMS...>::eval(b[i], e[i], x[i])) { return false; }}
        return true;
    }

};

template<>
struct value_in_range<>
{
    template<typename T0, typename T1, typename T2>
    static bool eval(T0 const &b, T1 const &e, T2 const &x) { return x >= b && x < e; }

};
}  // namespace _impl

template<size_type ... DIMS, typename T0, typename T1, typename T2>
bool value_in_range(T0 const &b, T1 const &e, T2 const &x)
{
    return _impl::value_in_range<DIMS...>::eval(b, e, x);
}

//template<typename T, size_type ...N>
//auto mod(nTuple<T, N...> const & l)
//DECL_RET_TYPE((std::sqrt(std::abs(inner_product(l,l)))))

#define _SP_DEFINE_nTuple_EXPR_BINARY_RIGHT_OPERATOR(_OP_, _NAME_)                                                  \
    template<typename T1,size_type ...N1,typename  T2> \
    constexpr nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>> \
    operator _OP_(nTuple<T1,N1...> const & l,T2 const &r)  \
    {return (nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>>(l,r)) ;}                 \


#define _SP_DEFINE_nTuple_EXPR_BINARY_OPERATOR(_OP_, _NAME_)                                                  \
    template<typename T1,size_type ...N1,typename  T2> \
    constexpr nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>> \
    operator _OP_(nTuple<T1, N1...> const & l,T2 const&r)  \
    {return (nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>>(l,r));}                    \
    \
    template< typename T1,typename T2 ,size_type ...N2> \
    constexpr nTuple<Expression< _impl::_NAME_,T1,nTuple< T2,N2...>>> \
    operator _OP_(T1 const & l, nTuple< T2,N2...>const &r)                    \
    {return (nTuple<Expression< _impl::_NAME_,T1,nTuple< T2,N2...>>>(l,r))  ;}                \
    \
    template< typename T1,size_type ... N1,typename T2 ,size_type ...N2>  \
    constexpr nTuple<Expression< _impl::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>\
    operator _OP_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
    {return (nTuple<Expression< _impl::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>(l,r));}                    \


#define _SP_DEFINE_nTuple_EXPR_UNARY_OPERATOR(_OP_, _NAME_)                           \
        template<typename T,size_type ...N> \
        constexpr nTuple<Expression<_impl::_NAME_,nTuple<T,N...> >> \
        operator _OP_(nTuple<T,N...> const &l)  \
        {return (nTuple<Expression<_impl::_NAME_,nTuple<T,N...> >>(l)) ;}    \


#define _SP_DEFINE_nTuple_EXPR_BINARY_FUNCTION(_NAME_)                                                  \
            template<typename T1,size_type ...N1,typename  T2> \
            constexpr    nTuple<BooleanExpression<_impl::_##_NAME_,nTuple<T1,N1...>,T2>> \
            _NAME_(nTuple<T1,N1...> const & l,T2 const &r)  \
            {return (nTuple<BooleanExpression<_impl::_##_NAME_,nTuple<T1,N1...>,T2>>(l,r));}       \
            \
            template< typename T1,typename T2,size_type ...N2> \
            constexpr    nTuple<Expression< _impl::_##_NAME_,T1,nTuple< T2,N2...>>>\
            _NAME_(T1 const & l, nTuple< T2,N2...>const &r)                    \
            {return (nTuple<Expression< _impl::_##_NAME_,T1,nTuple< T2,N2...>>>(l,r)) ;}       \
            \
            template< typename T1,size_type ... N1,typename T2,size_type  ...N2> \
            constexpr    nTuple<Expression< _impl::_##_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>\
            _NAME_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
            {return (nTuple<Expression< _impl::_##_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>(l,r))  ;}   \


#define _SP_DEFINE_nTuple_EXPR_UNARY_FUNCTION(_NAME_)                           \
        template<typename T,size_type ...N> \
        constexpr nTuple<Expression<_impl::_##_NAME_,nTuple<T,N...>>> \
        _NAME_(nTuple<T,N ...> const &r)  \
        {return (nTuple<Expression<_impl::_##_NAME_,nTuple<T,N...>>>(r));}     \


DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA2(nTuple)

/** @}*/



//////////////////////////////////////////////////////////////
/// @name reduction operation
//////////////////////////////////////////////////////////////
namespace traits
{

template<typename TOP, typename ...T>
struct primary_type<nTuple<BooleanExpression<TOP, T...> > > { typedef bool type; };
template<typename TOP, typename ...T>
struct pod_type<nTuple<BooleanExpression<TOP, T...> > > { typedef bool type; };
template<typename TOP, typename ...T>
struct extents<nTuple<BooleanExpression<TOP, T...> > > : public extents<nTuple<Expression<TOP, T...> > >
{
};

template<typename TOP, typename ...T>
struct value_type<nTuple<BooleanExpression<TOP, T...> > > { typedef bool type; };
}  // namespace traits

template<typename TOP, typename T> T const &reduce(TOP const &op, T const &v) { return v; }

template<typename TOP, typename T, size_type N0, size_type ...N>
traits::value_type_t<nTuple<T, N0, N...>> reduce(TOP const &op,
                                                 nTuple<T, N0, N...> const &v)
{
    static constexpr size_type n = N0;

    traits::value_type_t<nTuple<T, N0, N...> > res = reduce(op,
                                                            traits::index(v, 0));
    if (n > 1)
    {
        for (size_type s = 1; s < n; ++s)
        {
            res = op(res, reduce(op, traits::index(v, s)));
        }
    }
    return res;
}

template<typename TOP, typename ...T>
traits::value_type_t<nTuple<Expression<T...> > > reduce(TOP const &op,
                                                        nTuple<Expression<T...> > const &v)
{
    traits::primary_type_t<nTuple<Expression<T...> > > res = v;

    return reduce(op, res);
}

//template<typename TOP, typename ...Args>
//auto for_each(TOP const & op, Args &&... args)
//-> typename std::enable_if<!(mpl::logical_or<
//		traits::is_ntuple<Args>::m_value_...>::entity),void>::type
//{
//	op(std::forward<Args>(args)...);
//}

template<typename TOP, typename ...Args>
void for_each(TOP const &op, index_sequence<>, Args &&... args)
{
    op(std::forward<Args>(args) ...);
}

template<size_type N, size_type ...M, typename TOP, typename ...Args>
void for_each(TOP const &op, index_sequence<N, M...>,
              Args &&... args)
{
    for (size_type s = 0; s < N; ++s)
    {
        for_each(op, index_sequence<M...>(), traits::index(std::forward<Args>(args), s)...);
    }

}

template<typename TR, typename T, size_type ... N>
auto inner_product(nTuple<T, N...> const &l, TR const &r) { return (reduce(_impl::plus(), l * r)); }


inline constexpr double dot(double const &l, double const &r) { return r * l; }

inline constexpr float dot(float const &l, float const &r) { return r * l; }

template<typename TR, typename T, size_type ... N>
auto dot(nTuple<T, N...> const &l, TR const &r) { return (inner_product(l, r)); }

template<typename T, size_type ... N>
auto normal(nTuple<T, N...> const &l) { return std::sqrt(inner_product(l, l)); }

template<typename T> auto sp_abs(T const &v) { return std::abs(v); }

template<typename T, size_type ...N>
auto sp_abs(nTuple<T, N...> const &m) { return std::sqrt(std::abs(inner_product(m, m))); }

template<typename ... T>
auto sp_abs(nTuple<Expression<T...>> const &m) { return std::sqrt(std::abs(inner_product(m, m))); }

template<typename T> auto mod(T const &v) { return sp_abs(v); }

template<typename T, size_type ...N>
auto abs(nTuple<T, N...> const &v) { return sp_abs(v); }

template<typename T, size_type ...N>
inline auto NProduct(nTuple<T, N...> const &v) { return reduce(_impl::multiplies(), v); }

template<typename T, size_type ...N>
inline auto NSum(nTuple<T, N...> const &v) { return reduce(_impl::plus(), v); }

template<typename TOP, typename ... T>
struct nTuple<BooleanExpression<TOP, T...> > : public nTuple<
        Expression<TOP, T...> >
{
    typedef nTuple<BooleanExpression<TOP, T...>>
            this_type;

    using nTuple<Expression<TOP, T...>>::at;
    using nTuple<Expression<TOP, T...>>::nTuple;

    operator bool() const
    {
        static constexpr size_type N = mpl::max<int,
                traits::extent<T, 0>::type::value...>::value;

        bool res = static_cast<bool>(at(0));

        if (N > 1)
        {
            for (size_type s = 1; s < N; ++s)
            {
                res = typename _impl::op_traits<TOP>::type()(res,
                                                             static_cast<bool>(at(s)));
            }
        }
        return res;
    }

};

#define _SP_DEFINE_nTuple_EXPR_BINARY_BOOLEAN_OPERATOR(_OP_, _NAME_)                                                  \
    template<typename T1,size_type ...N1,typename  T2> \
    constexpr nTuple<BooleanExpression<_impl::_NAME_,nTuple<T1,N1...>,T2>> \
    operator _OP_(nTuple<T1, N1...> const & l,T2 const&r)  \
    {return (nTuple<BooleanExpression<_impl::_NAME_,nTuple<T1,N1...>,T2>>(l,r));}                    \
    \
    template< typename T1,typename T2 ,size_type ...N2> \
    constexpr nTuple<BooleanExpression< _impl::_NAME_,T1,nTuple< T2,N2...>>> \
    operator _OP_(T1 const & l, nTuple< T2,N2...>const &r)                    \
    {return (nTuple<BooleanExpression< _impl::_NAME_,T1,nTuple< T2,N2...>>>(l,r))  ;}                \
    \
    template< typename T1,size_type ... N1,typename T2 ,size_type ...N2>  \
    constexpr nTuple<BooleanExpression< _impl::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>\
    operator _OP_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
    {return (nTuple<BooleanExpression< _impl::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>(l,r));}                    \


#define _SP_DEFINE_nTuple_EXPR_UNARY_BOOLEAN_OPERATOR(_OP_, _NAME_)                           \
        template<typename T,size_type ...N> \
        constexpr nTuple<BooleanExpression<_impl::_NAME_,nTuple<T,N...> >> \
        operator _OP_(nTuple<T,N...> const &l)  \
        {return (nTuple<BooleanExpression<_impl::_NAME_,nTuple<T,N...> >>(l)) ;}    \


DEFINE_EXPRESSOPM_TEMPLATE_BOOLEAN_ALGEBRA2(nTuple)

}
//namespace simpla

namespace std
{

template<typename T, size_type N, size_type ... M>
void swap(simpla::nTuple<T, N, M...> &l, simpla::nTuple<T, N, M...> &r)
{
    for (size_type s = 0; s < N; ++s)
    {
        swap(simpla::traits::index(l, s), simpla::traits::index(r, s));
    }
}

template<typename T, size_type N, size_type ... M>
void swap(simpla::nTuple<T, N, M...> &l,
          simpla::traits::pod_type_t<simpla::nTuple<T, N, M...>> &r)
{

    for (size_type s = 0; s < N; ++s)
    {
        swap(simpla::traits::index(l, s), simpla::traits::index(r, s));
    }
}

}

#endif  // CORE_toolbox_NTUPLE_H_
