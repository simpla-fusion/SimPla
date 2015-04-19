/**
 * Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id: nTuple.h 990 2010-12-14 11:06:21Z salmon $
 * @file ntuple.h
 *
 *  created on: Jan 27, 2010
 *      Author: yuzhi
 */

#ifndef CORE_GTL_NTUPLE_H_
#define CORE_GTL_NTUPLE_H_

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "expression_template.h"
#include "integer_sequence.h"
#include "type_traits.h"

namespace simpla
{

/**
 * @ingroup gtl
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
 *   template<typename T, size_t ... n> struct nTuple;
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
template<typename, size_t...>
struct nTuple;
template<typename>
struct nTuple_traits;

template<typename TV>
struct nTuple<TV>
{
    typedef TV value_type;

    typedef void sub_type;

    typedef integer_sequence<size_t> dimensions;

    typedef value_type pod_type;

};


template<typename TV, size_t N>
struct nTuple<TV, N>
{

    typedef TV value_type;

    typedef value_type sub_type;

    typedef integer_sequence<size_t, N> dimensions;

    typedef value_type pod_type[N];

    static constexpr size_t dims = N;

    typedef nTuple<value_type, N> this_type;

    sub_type data_[dims];

    sub_type &operator[](size_t s)
    {
        return data_[s];
    }

    sub_type const &operator[](size_t s) const
    {
        return data_[s];
    }

    this_type &operator++()
    {
        ++data_[N - 1];
        return *this;
    }

    this_type &operator--()
    {
        --data_[N - 1];
        return *this;
    }

private:
    template<size_t N1, size_t N2>
    struct min_not_zero
    {
        static constexpr size_t value = ((N2 < N1) && N2 != 0) ? N2 : N1;
    };
public:

    template<typename TR>
    inline this_type &
    operator=(TR const &rhs)
    {
        _seq_for<dims>::eval(_impl::_assign(), data_, rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &
    operator=(TR const *rhs)
    {
        _seq_for<dims>::eval(_impl::_assign(), data_, rhs);

        return (*this);
    }

//	template<typename TR>
//	inline bool operator ==(TR const &rhs)
//	{
//		return _seq_reduce<
//				min_not_zero<dims,
//						seq_get<0, typename nTuple_traits<TR>::dimensions>::value>::value>::eval(
//				_impl::logical_and(), _impl::equal_to(), data_, rhs);;
//	}
//
    template<typename TR>
    inline this_type &operator+=(TR const &rhs)
    {
        _seq_for<dims>::eval(_impl::plus_assign(), data_, rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator-=(TR const &rhs)
    {
        _seq_for<dims>::eval(_impl::minus_assign(), data_, rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator*=(TR const &rhs)
    {
        _seq_for<dims>::eval(_impl::multiplies_assign(), data_, rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator/=(TR const &rhs)
    {
        _seq_for<dims>::eval(_impl::divides_assign(), data_, rhs);
        return (*this);
    }

//	template<size_t NR, typename TR>
//	void operator*(nTuple<NR, TR> const & rhs) = delete;
//
//	template<size_t NR, typename TR>
//	void operator/(nTuple<NR, TR> const & rhs) = delete;

};

template<typename TV, size_t N, size_t ...M>
struct nTuple<TV, N, M...>
{

    typedef TV value_type;

    typedef typename std::conditional<(sizeof...(M) == 0), value_type,
            nTuple<value_type, M...>>::type sub_type;

    typedef integer_sequence<size_t, N, M...> dimensions;

    typedef typename std::conditional<(sizeof...(M) == 0), value_type,
            typename nTuple<value_type, M...>::pod_type>::type pod_type[N];

    static constexpr size_t dims = seq_get<0, dimensions>::value;

    typedef nTuple<value_type, N, M...> this_type;

    sub_type data_[dims];

    sub_type &operator[](size_t s)
    {
        return data_[s];
    }

    sub_type const &operator[](size_t s) const
    {
        return data_[s];
    }

    this_type &operator++()
    {
        ++data_[N - 1];
        return *this;
    }

    this_type &operator--()
    {
        --data_[N - 1];
        return *this;
    }

private:
    template<size_t N1, size_t N2>
    struct min_not_zero
    {
        static constexpr size_t value = ((N2 < N1) && N2 != 0) ? N2 : N1;
    };
public:

    template<typename TR>
    inline this_type &
    operator=(TR const &rhs)
    {

        //  assign different 'dimensions' ntuple
        _seq_for<
                min_not_zero<dims,
                        seq_get<0, typename nTuple_traits<TR>::dimensions>::value>::value

        >::eval(_impl::_assign(), data_, rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &
    operator=(TR const *rhs)
    {
        _seq_for<dims>::eval(_impl::_assign(), data_, rhs);

        return (*this);
    }

//	template<typename TR>
//	inline bool operator ==(TR const &rhs)
//	{
//		return _seq_reduce<
//				min_not_zero<dims,
//						seq_get<0, typename nTuple_traits<TR>::dimensions>::value>::value>::eval(
//				_impl::logical_and(), _impl::equal_to(), data_, rhs);;
//	}
//
    template<typename TR>
    inline this_type &operator+=(TR const &rhs)
    {
        _seq_for<
                min_not_zero<dims,
                        seq_get<0, typename nTuple_traits<TR>::dimensions>::value>::value>::eval(
                _impl::plus_assign(), data_, rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator-=(TR const &rhs)
    {
        _seq_for<
                min_not_zero<dims,
                        seq_get<0, typename nTuple_traits<TR>::dimensions>::value>::value>::eval(
                _impl::minus_assign(), data_, rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator*=(TR const &rhs)
    {
        _seq_for<
                min_not_zero<dims,
                        seq_get<0, typename nTuple_traits<TR>::dimensions>::value>::value>::eval(
                _impl::multiplies_assign(), data_, rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator/=(TR const &rhs)
    {
        _seq_for<
                min_not_zero<dims,
                        seq_get<0, typename nTuple_traits<TR>::dimensions>::value>::value>::eval(
                _impl::divides_assign(), data_, rhs);
        return (*this);
    }

//	template<size_t NR, typename TR>
//	void operator*(nTuple<NR, TR> const & rhs) = delete;
//
//	template<size_t NR, typename TR>
//	void operator/(nTuple<NR, TR> const & rhs) = delete;

};

template<typename T1, typename ...T>
nTuple<T1, 1 + sizeof...(T)> make_nTuple(T1 &&a1, T &&... a)
{
    return std::move(nTuple<T1, 1 + sizeof...(T)>(
            {std::forward<T1>(a1), std::forward<T>(a)...}));
}

template<typename ...>
struct make_pod_array;

template<typename TV, typename TI, TI ... N>
struct make_pod_array<TV, integer_sequence<TI, N...>>
{
    typedef typename nTuple<TV, N...>::pod_type type;
};

template<typename ...>
struct nTuple_create_trait;
template<typename TV, typename TI, TI ... N>
struct nTuple_create_trait<TV, integer_sequence<TI, N...>>
{
    typedef nTuple<TV, N...> type;
};

template<typename TV, typename TI>
struct nTuple_create_trait<TV, integer_sequence<TI>>
{
    typedef TV type;
};
/**
 * @brief Convert fixed size build-in array to nTuple
 *
 * Example:
 *  typename array_to_ntuple_convert<double[3][4]>::type = nTuple<double,3,4>
 */
template<typename T>
struct array_to_ntuple_convert
{
    typedef integer_sequence<size_t> extents_t;

    typedef T type;
};
template<typename T, size_t N>
struct array_to_ntuple_convert<T[N]>
{

    typedef typename cat_integer_sequence<
            typename array_to_ntuple_convert<T>::extents_t,
            integer_sequence<size_t, N>>::type extents_t;

    typedef typename nTuple_create_trait<
            typename std::remove_all_extents<T>::type, extents_t>::type type;
};

template<typename ...>
class Expression;

template<typename>
struct nTuple_traits;

template<typename ... T>
struct nTuple<Expression<T...>> : public Expression<T...>
{
    typedef nTuple<Expression<T...>> this_type;

    using Expression<T...>::Expression;

    typedef typename nTuple_traits<this_type>::primary_type primary_type;

    operator primary_type() const
    {
        primary_type res;
        res = *this;
        return std::move(res);
    }

};

template<typename TOP, typename ... T>
struct nTuple<BooleanExpression<TOP, T...>> : public Expression<TOP, T...>
{
    typedef nTuple<BooleanExpression<TOP, T...>> this_type;

    using Expression<TOP, T...>::Expression;

    operator bool() const
    {
        return seq_reduce(typename nTuple_traits<this_type>::dimensions(),
                          typename _impl::op_traits<TOP>::reduction_op(), *this);
    }

};

template<typename>
struct reference_traits;

template<typename T, size_t M, size_t ...N>
struct reference_traits<nTuple<T, M, N...>>
{
    typedef nTuple<T, M, N...> const &type;
};

template<typename ...T>
struct reference_traits<nTuple<Expression<T...> >>
{
    typedef nTuple<Expression<T...> > type;
};

template<typename TV>
struct nTuple_traits
{
    typedef integer_sequence<size_t> dimensions;

    static constexpr size_t ndims = 0;
    static constexpr size_t first_dims = 0;

    typedef TV value_type;
    typedef TV primary_type;

};

template<typename>
struct is_ntuple
{
    static constexpr bool value = false;
};

template<typename T, size_t ...N>
struct is_ntuple<nTuple<T, N...>>
{
    static constexpr bool value = true;
};

template<typename TV, size_t N, size_t ...M>
struct nTuple_traits<nTuple<TV, N, M...> >
{
    static constexpr size_t ndims = 1 + sizeof...(M);

    typedef typename nTuple_traits<TV>::value_type value_type;

    typedef typename cat_integer_sequence<integer_sequence<size_t, N, M...>,
            typename nTuple_traits<TV>::dimensions>::type dimensions;

    typedef typename make_pod_array<value_type, dimensions>::type pod_type;

    typedef typename nTuple_create_trait<value_type, dimensions>::type primary_type;

};

template<typename TOP, typename TL>
struct nTuple_traits<nTuple<Expression<TOP, TL, std::nullptr_t> > >
{
private:
    typedef typename nTuple_traits<TL>::dimensions d_seq_l;
    typedef typename nTuple_traits<TL>::value_type value_type_l;
public:
    typedef d_seq_l dimensions;

    typedef decltype(std::declval<TOP>()(std::declval<value_type_l>())) value_type;

    typedef typename make_pod_array<value_type, dimensions>::type pod_type;

    typedef typename nTuple_create_trait<value_type, dimensions>::type primary_type;

};
template<typename TOP, typename TL, typename TR>
struct nTuple_traits<nTuple<Expression<TOP, TL, TR>>>
{
private:
    typedef typename nTuple_traits<TL>::dimensions d_seq_l;
    typedef typename nTuple_traits<TR>::dimensions d_seq_r;
    typedef typename nTuple_traits<TL>::value_type value_type_l;
    typedef typename nTuple_traits<TR>::value_type value_type_r;
public:
    typedef typename longer_integer_sequence<d_seq_l, d_seq_r>::type dimensions;

    typedef decltype(std::declval<TOP>()(std::declval<value_type_l>(),
                                         std::declval<value_type_r>())) value_type;

    typedef typename make_pod_array<value_type, dimensions>::type pod_type;

    typedef typename nTuple_create_trait<value_type, dimensions>::type primary_type;

};

template<typename TOP, typename ...T>
struct nTuple_traits<nTuple<BooleanExpression<TOP, T...>>>
{

    typedef typename nTuple_traits<nTuple<Expression<TOP, T...>>>::dimensions dimensions;

    typedef bool value_type;

    typedef bool pod_type;

    typedef bool primary_type;

};

template<typename T, size_t ...N>
struct sp_pod_traits<nTuple<T, N...> >
{
    typedef typename nTuple_traits<nTuple<T, N...>>::primary_type type;

};

//template<typename T, size_t ...N>
//struct rank<nTuple<T, N...>>
//{
//	static constexpr size_t value =
//			nTuple_traits<nTuple<T, N...>>::dimensions::size();
//};

template<typename TInts, TInts ...N>
nTuple<TInts, sizeof...(N)> seq2ntuple(integer_sequence<TInts, N...>)
{
    return std::move(nTuple<TInts, sizeof...(N)>({N...}));
}

template<typename TV, size_t N, typename T1>
nTuple<TV, N> append_ntuple(T1 const &v0, TV const &v1)
{
    nTuple<TV, N> res;
    res = v0;
    res[N - 1] = v1;
    return std::move(res);
}

template<typename TV, size_t N, typename T2>
nTuple<TV, N + 1> join_ntuple(nTuple<TV, N> const &left, T2 right)
{
    nTuple<TV, N + 1> res;
    res = left;
    res[N] = right;
    return std::move(res);
}

template<typename T1, size_t N, typename T2, size_t M>
nTuple<T1, N + M> join_ntuple(nTuple<T1, N> const &left, nTuple<T2, M> right)
{
    nTuple<T1, N + M> res;
    for (int i = 0; i < N; ++i) {
        res[i] = left[i];
    }
    for (int i = 0; i < M; ++i) {
        res[i + N] = right[i];
    }
    return std::move(res);
}

template<typename T, size_t N> using Vector=nTuple<T, N>;

template<typename T, size_t M, size_t N> using Matrix=nTuple<T, M, N>;

template<typename T, size_t ... N> using Tensor=nTuple<T, N...>;

template<typename T, size_t N, size_t ... M>
void swap(nTuple<T, N, M...> &l, nTuple<T, N, M...> &r)
{
    _seq_for<N>::eval(_impl::_swap(), (l), (r));
}

template<typename T, size_t N, size_t ... M>
void swap(nTuple<T, N, M...> &l,
          typename nTuple_traits<nTuple<T, N, M...>>::pod_type &r)
{
    _seq_for<N>::eval(_impl::_swap(), (l), (r));
}

template<typename TR, typename T, size_t ... N>
void assign(nTuple<T, N...> &l, TR const &r)
{
    _seq_for<N...>::eval(_impl::_assign(), l, r);
}

template<typename TR, typename T, size_t ... N>
auto inner_product(nTuple<T, N...> const &l, TR const &r)
DECL_RET_TYPE ((_seq_reduce<N...>::eval(_impl::plus(), l * r)))

template<typename TR, typename T, size_t ... N>
auto dot(nTuple<T, N...> const &l, TR const &r)
DECL_RET_TYPE ((_seq_reduce<N...>::eval(_impl::plus(), l * r)))

template<typename T, size_t ... N>
auto normal(
        nTuple<T, N...> const &l)
DECL_RET_TYPE((std::sqrt((_seq_reduce<N...>::eval(_impl::plus(), l * l)))))

template<typename TR, typename ...T>
auto inner_product(nTuple<Expression<T...>> const &l,
                   TR const &r)
DECL_RET_TYPE ((seq_reduce(typename nTuple_traits<nTuple<Expression<T...>>>::dimensions(),
                           _impl::plus(), l * r))
)

template<typename T, size_t M, size_t ... N>
double mod(nTuple<T, M, N...> const &l)
{
    return std::sqrt(inner_product(l, l));
}

template<typename ...T>
double mod(nTuple<Expression<T...>> l)
{
    return std::sqrt(std::abs(inner_product(l, l)));
}

inline double mod(double const &v)
{
    return std::abs(v);
}

template<typename TR, typename T, size_t ... N>
auto dot(nTuple<T, N...> const &l, TR const &r)
DECL_RET_TYPE ((inner_product(l, r)))

template<typename T>
inline auto determinant(nTuple<T, 3> const &m)
DECL_RET_TYPE(m[0] * m[1] * m[2])

template<typename T>
inline auto determinant(nTuple<T, 4> const &m)
DECL_RET_TYPE(m[0] * m[1] * m[2] * m[3])

template<typename T>
inline auto determinant(
        Matrix<T, 3, 3> const &m)
DECL_RET_TYPE((m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] //
                                                                           * m[1][2] * m[2][0] -
               m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1]//
                                             * m[0][2] - m[1][2] * m[2][1] * m[0][0]//
              )

)

template<typename T>
inline auto determinant(
        Matrix<T, 4, 4> const &m)
DECL_RET_TYPE((//
                      m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1] * m[3][0]//
                      - m[0][3] * m[1][1] * m[2][2] * m[3][0] + m[0][1] * m[1][3]//
                                                                * m[2][2] * m[3][0] +
                      m[0][2] * m[1][1] * m[2][3] * m[3][0] - m[0][1]//
                                                              * m[1][2] * m[2][3] * m[3][0] -
                      m[0][3] * m[1][2] * m[2][0] * m[3][1]//
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
              ))

template<typename T>
auto sp_abs(T const &v)
DECL_RET_TYPE((std::abs(v)))

template<typename T, size_t ...N>
auto sp_abs(nTuple<T, N...> const &m)
DECL_RET_TYPE((std::sqrt(std::abs(inner_product(m, m)))))

template<typename T, size_t ...N>
inline
auto NProduct(nTuple<T, N...> const &v)
DECL_RET_TYPE((seq_reduce(
        typename nTuple_traits<nTuple<T, N...>>::dimensions(),
        _impl::multiplies(), v)))

template<typename T, size_t ...N>
inline
auto NSum(nTuple<T, N...> const &v)
DECL_RET_TYPE((seq_reduce(
        typename nTuple_traits<nTuple<T, N...>>::dimensions(),
        _impl::plus(), v)))

template<typename T1, size_t ... N1, typename T2, size_t ... N2>
inline auto cross(
        nTuple<T1, N1...> const &l, nTuple<T2, N2...> const &r)
-> nTuple<decltype(get_value(l, 0) * get_value(r, 0)), 3>
{
    nTuple<decltype(get_value(l, 0) * get_value(r, 0)), 3> res = {l[1] * r[2]
                                                                  - l[2] * r[1],
                                                                  l[2] * get_value(r, 0) - get_value(l, 0) * r[2],
                                                                  get_value(l, 0) * r[1] - l[1] * get_value(r, 0)};
    return std::move(res);
}

inline nTuple<double, 3> cross(nTuple<double, 3> const &l,
                               nTuple<double, 3> const &r)
{
    return std::move(
            nTuple<double, 3>(
                    {l[1] * r[2] - l[2] * r[1], l[2] * r[0] - l[0] * r[2], l[0]
                                                                           * r[1] - l[1] * r[0]}));
}

template<typename T>
auto mod(T const &l)
DECL_RET_TYPE ((abs(l)))

//template<typename T, size_t ...N>
//auto mod(nTuple<T, N...> const & l)
//DECL_RET_TYPE((std::sqrt(std::abs(inner_product(l,l)))))

#define _SP_DEFINE_nTuple_EXPR_BINARY_RIGHT_OPERATOR(_OP_, _NAME_)                                                  \
    template<typename T1,size_t ...N1,typename  T2> \
    constexpr nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>> \
    operator _OP_(nTuple<T1,N1...> const & l,T2 const &r)  \
    {return (nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>>(l,r)) ;}                 \


#define _SP_DEFINE_nTuple_EXPR_BINARY_OPERATOR(_OP_, _NAME_)                                                  \
    template<typename T1,size_t ...N1,typename  T2> \
    constexpr nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>> \
    operator _OP_(nTuple<T1, N1...> const & l,T2 const&r)  \
    {return (nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>>(l,r));}                    \
    \
    template< typename T1,typename T2 ,size_t ...N2> \
    constexpr nTuple<Expression< _impl::_NAME_,T1,nTuple< T2,N2...>>> \
    operator _OP_(T1 const & l, nTuple< T2,N2...>const &r)                    \
    {return (nTuple<Expression< _impl::_NAME_,T1,nTuple< T2,N2...>>>(l,r))  ;}                \
    \
    template< typename T1,size_t ... N1,typename T2 ,size_t ...N2>  \
    constexpr nTuple<Expression< _impl::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>\
    operator _OP_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
    {return (nTuple<Expression< _impl::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>(l,r));}                    \


#define _SP_DEFINE_nTuple_EXPR_UNARY_OPERATOR(_OP_, _NAME_)                           \
        template<typename T,size_t ...N> \
        constexpr nTuple<Expression<_impl::_NAME_,nTuple<T,N...> , std::nullptr_t>> \
        operator _OP_(nTuple<T,N...> const &l)  \
        {return (nTuple<Expression<_impl::_NAME_,nTuple<T,N...> , std::nullptr_t>>(l)) ;}    \


#define _SP_DEFINE_nTuple_EXPR_BINARY_BOOLEAN_OPERATOR(_OP_, _NAME_)                                                  \
    template<typename T1,size_t ...N1,typename  T2> \
    constexpr nTuple<BooleanExpression<_impl::_NAME_,nTuple<T1,N1...>,T2>> \
    operator _OP_(nTuple<T1, N1...> const & l,T2 const&r)  \
    {return (nTuple<BooleanExpression<_impl::_NAME_,nTuple<T1,N1...>,T2>>(l,r));}                    \
    \
    template< typename T1,typename T2 ,size_t ...N2> \
    constexpr nTuple<BooleanExpression< _impl::_NAME_,T1,nTuple< T2,N2...>>> \
    operator _OP_(T1 const & l, nTuple< T2,N2...>const &r)                    \
    {return (nTuple<BooleanExpression< _impl::_NAME_,T1,nTuple< T2,N2...>>>(l,r))  ;}                \
    \
    template< typename T1,size_t ... N1,typename T2 ,size_t ...N2>  \
    constexpr nTuple<BooleanExpression< _impl::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>\
    operator _OP_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
    {return (nTuple<BooleanExpression< _impl::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>(l,r));}                    \


#define _SP_DEFINE_nTuple_EXPR_UNARY_BOOLEAN_OPERATOR(_OP_, _NAME_)                           \
        template<typename T,size_t ...N> \
        constexpr nTuple<BooleanExpression<_impl::_NAME_,nTuple<T,N...> , std::nullptr_t>> \
        operator _OP_(nTuple<T,N...> const &l)  \
        {return (nTuple<BooleanExpression<_impl::_NAME_,nTuple<T,N...> , std::nullptr_t>>(l)) ;}    \


#define _SP_DEFINE_nTuple_EXPR_BINARY_FUNCTION(_NAME_)                                                  \
            template<typename T1,size_t ...N1,typename  T2> \
            constexpr    nTuple<BooleanExpression<_impl::_##_NAME_,nTuple<T1,N1...>,T2>> \
            _NAME_(nTuple<T1,N1...> const & l,T2 const &r)  \
            {return (nTuple<BooleanExpression<_impl::_##_NAME_,nTuple<T1,N1...>,T2>>(l,r));}       \
            \
            template< typename T1,typename T2,size_t ...N2> \
            constexpr    nTuple<Expression< _impl::_##_NAME_,T1,nTuple< T2,N2...>>>\
            _NAME_(T1 const & l, nTuple< T2,N2...>const &r)                    \
            {return (nTuple<Expression< _impl::_##_NAME_,T1,nTuple< T2,N2...>>>(l,r)) ;}       \
            \
            template< typename T1,size_t ... N1,typename T2,size_t  ...N2> \
            constexpr    nTuple<Expression< _impl::_##_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>\
            _NAME_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
            {return (nTuple<Expression< _impl::_##_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>(l,r))  ;}   \


#define _SP_DEFINE_nTuple_EXPR_UNARY_FUNCTION(_NAME_)                           \
        template<typename T,size_t ...N> \
        constexpr nTuple<Expression<_impl::_##_NAME_,nTuple<T,N...>, std::nullptr_t>> \
        _NAME_(nTuple<T,N ...> const &r)  \
        {return (nTuple<Expression<_impl::_##_NAME_,nTuple<T,N...>, std::nullptr_t>>(r));}     \


DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA2(nTuple)

/** @}*/

}
//namespace simpla

namespace std
{
/**
 * C++11 <type_traits>
 * @ref http://en.cppreference.com/w/cpp/types/rank
 */
template<typename T, size_t ...N>
struct rank<simpla::nTuple<T, N...>> : public std::integral_constant<
        std::size_t, sizeof...(N)>
{
};

/**
 * C++11 <type_traits>
 * @ref http://en.cppreference.com/w/cpp/types/extent
 */

template<class T, std::size_t N, std::size_t ...M>
struct extent<simpla::nTuple<T, N, M...>, 0> : std::integral_constant<
        std::size_t, N>
{
};

template<std::size_t I, class T, std::size_t N, std::size_t ...M>
struct extent<simpla::nTuple<T, N, M...>, I> : public std::integral_constant<
        std::size_t, std::extent<simpla::nTuple<T, M...>, I - 1>::value>
{
};

/**
 * C++11 <type_traits>
 * @ref http://en.cppreference.com/w/cpp/types/remove_all_extents
 */
template<class T, std::size_t ...M>
struct remove_all_extents<simpla::nTuple<T, M...> >
{
    typedef T type;
};

//template<typename T, size_t I>
//class std::less<simpla::nTuple<T, I> >
//{
//public:
//	bool operator()(const simpla::nTuple<T, I>& x,
//			const simpla::nTuple<T, I>& y) const
//	{
//		for (int i = 0; i < I; ++i)
//		{
//			if (x[i] < y[i])
//				return true;
//		}
//		return false;
//	}
//};
}// namespace std
#endif  // CORE_GTL_NTUPLE_H_
