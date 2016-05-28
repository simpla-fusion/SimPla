/** 
 * @file nTupleLite.h
 * @author salmon
 * @date 16-5-27 - 上午8:56
 *  */

#ifndef SIMPLA_NTUPLELITE_H
#define SIMPLA_NTUPLELITE_H


#include <cstdbool>
#include <cstddef>
#include <type_traits>

#include "macro.h"
#include "mpl.h"
#include "type_traits.h"
#include "integer_sequence.h"
#include "port_cxx14.h"

#include "ExpressionTemplate.h"

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
template<typename, size_t ...> struct nTuple;

template<typename ...> class Expression;


template<typename TV> struct nTuple<TV>
{
private:


    typedef nTuple<TV> this_type;


public:

    typedef TV value_type;

    value_type data_;

    value_type &at(size_t) { return data_; }

    value_type const &at(size_t) const { return data_; }

    value_type &operator[](size_t s) { return at(s); }

    value_type const &operator[](size_t s) const { return at(s); }


public:


    template<typename Op, typename TR>
    void assign(Op const &op, TR const &rhs) { op(data_, traits::index(rhs, 0)); }

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

template<typename TV, size_t N>
struct nTuple<TV, N>
{
private:

    typedef TV value_type;

    typedef nTuple<value_type, N> this_type;


public:
    value_type data_[N];

    value_type &operator[](size_t s) { return data_[s]; }

    value_type const &operator[](size_t s) const { return data_[s]; }


    value_type &at(size_t s) { return data_[s]; }

    value_type const &at(size_t s) const { return data_[s]; }

private:


    template<typename Op, typename TR>
    void assign(Op const &op, TR const &rhs) { for (size_t s = 0; s < N; ++s) { op(data_[s], traits::index(rhs, s)); }}

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

template<typename TOP, typename ... Args>
struct nTuple<Expression<TOP, Args...>> : public Expression<TOP, Args...>
{
    typedef nTuple<Expression<TOP, Args...>> this_type;

    using Expression<TOP, Args...>::m_op_;
    using Expression<TOP, Args...>::args;
    using Expression<TOP, Args...>::Expression;

    static constexpr size_t _N = traits::extent<this_type>::value;//std::max<traits::extent<Args>::value...>::value;

    template<typename U, size_t ..._I>
    operator nTuple<U, _I...>() const
    {
        nTuple<U, _I...> res;
        res = *this;
        return std::move(res);
    }


    template<typename ID>
    inline auto operator[](ID const &s) const DECL_RET_TYPE (at(s))

private:
    template<typename ID, size_t ... index>
    auto _invoke_helper(ID s, index_sequence<index...>) const DECL_RET_TYPE(
            m_op_(traits::index(std::get<index>(args), s)...))

public:
    template<typename ID>
    auto at(ID const &s) const
    -> decltype(_invoke_helper(s, index_sequence_for<Args...>()))
    {
        return (_invoke_helper(s, index_sequence_for<Args...>()));
    }


    template<typename ID>
    inline auto operator[](ID const &s) const
    -> decltype(_invoke_helper(s, index_sequence_for<Args...>()))
    {
        return (_invoke_helper(s, index_sequence_for<Args...>()));
    }


};
//

template<typename TOP, typename ... T>
struct nTuple<BooleanExpression<TOP, T...> >
        : public nTuple<Expression<TOP, T...> >
{
    typedef nTuple<BooleanExpression<TOP, T...>>
            this_type;

    using nTuple<Expression<TOP, T...>>::at;
    using nTuple<Expression<TOP, T...>>::nTuple;

    operator bool() const
    {
        static constexpr size_t N = mpl::max<int,
                traits::extent<T, 0>::type::value...>::value;

        bool res = static_cast<bool>(at(0));

        if (N > 1)
        {
            for (size_t s = 1; s < N; ++s)
            {
                res = typename _impl::op_traits<TOP>::type()(res,
                                                             static_cast<bool>(at(s)));
            }
        }
        return res;
    }

};

//----------------------------------------------------------------------------------------------------------------------
namespace traits
{

template<typename> struct is_ntuple { static constexpr bool value = false; };

template<typename T, size_t ...N> struct is_ntuple<nTuple<T, N...>> { static constexpr bool value = true; };

template<typename T, size_t N> struct key_type<nTuple<T, N >> { typedef size_t type; };

template<typename TV, size_t N>
struct value_type<nTuple<TV, N> > { typedef traits::value_type_t<TV> type; };


template<typename TOP, typename ... T>
struct value_type<nTuple<Expression<TOP, T...> > >
{
    typedef std::result_of_t<TOP(traits::value_type_t<T>...)> type;
};
//----------------------------------------------------------------------------------------------------------------------

template<typename T> struct rank;

template<typename T, size_t N> struct rank<nTuple<T, N>> : public index_const<rank<T>::value + 1> { };

template<typename TV, size_t N> struct extent<TV, N> : public index_const<0> { };

template<typename TV, size_t M, size_t N> struct extent<nTuple<TV, M>, N> : public extent<TV, N + 1> { };

template<typename TV, size_t M> struct extent<nTuple<TV, M>, 0> : public index_const<M> { };


//----------------------------------------------------------------------------------------------------------------------

namespace _impl
{
template<typename ...> struct expr_extents_helper;
template<typename ...> struct longest_seq;


/**
 * integer sequence of the number of element along all dimensions
 * i.e.
 *
 */
template<typename> struct make_extents { typedef index_sequence<> type; };

template<typename T, int N>
struct make_extents<T[N]>
{
    typedef traits::seq_concat<index_sequence<N>, typename make_extents<T>::type> type;
};


template<typename T, size_t N>
struct make_extents<nTuple<T, N>>
{
    typedef typename traits::seq_concat<index_sequence<N>, typename make_extents<T>::type> type;
};

template<typename TOP, typename ... T>
struct make_extents<nTuple<Expression<TOP, T...> > >
{
    typedef typename expr_extents_helper<Expression<TOP, T...>>::type type;
};

template<size_t ...N>
struct longest_seq<index_sequence<N...>, index_sequence<> >
{
    typedef index_sequence<N...> type;
};

template<size_t ...N>
struct longest_seq<index_sequence<>, index_sequence<N...> >
{
    typedef index_sequence<N...> type;
};


template<size_t ...N, size_t ...M>
struct longest_seq<index_sequence<N...>, index_sequence<M...> >
{


    typedef std::conditional_t<(sizeof...(N) < sizeof...(M)), index_sequence<N...>,
            index_sequence<M...> > type;
};

template<typename TOP>
struct expr_extents_helper<Expression<TOP >>
{
    typedef index_sequence<> type;
};
template<typename TOP, typename First>
struct expr_extents_helper<Expression<TOP, First> >
{
    typedef typename make_extents<First>::type type;
};
template<typename TOP, typename First, typename ...Others>
struct expr_extents_helper<Expression<TOP, First, Others...>>
{
    typedef typename longest_seq<typename make_extents<First>::type,
            typename make_extents<Others>::type...>::type type;
};


}// namespace _impl
template<typename T> using extents=typename _impl::make_extents<T>::type;


//----------------------------------------------------------------------------------------------------------------------
template<typename> struct primary_type;

template<typename> struct pod_type;

namespace _impl
{

template<typename ...> struct make_pod_array;
template<typename ...> struct make_primary_nTuple;

template<typename TV, typename TI> struct make_pod_array<TV, integer_sequence<TI>>
{
    typedef TV type;
};
template<typename TV, typename TI, TI N0, TI ... N>
struct make_pod_array<TV, integer_sequence<TI, N0, N...>>
{
    typedef typename make_pod_array<TV, integer_sequence<TI, N...>>
    ::type type[N0];
};

template<typename TV, typename TI, TI ... N>
struct make_primary_nTuple<TV, integer_sequence<TI, N...>>
{
    typedef nTuple<TV, N...> type;
};
template<typename TV, typename TI>
struct make_primary_nTuple<TV, integer_sequence<TI>>
{
    typedef TV type;
};

template<typename ... T> using make_pod_array_t = typename make_pod_array<T...>::type;
template<typename ... T> using make_primary_nTuple_t = typename make_primary_nTuple<T...>::type;


}// namespace _impl

template<typename T, int ...N>
struct primary_type<nTuple<T, N...>>
{
    typedef _impl::make_primary_nTuple_t<

            traits::value_type_t<nTuple<T, N...>>,

            traits::extents<nTuple<T, N...>>
    >
            type;

};

template<typename T> using ntuple_cast_t=typename primary_type<T>::type;

template<typename T, size_t ...N>
struct pod_type<nTuple<T, N...>>
{
    typedef _impl::make_pod_array_t<

            traits::value_type_t<nTuple<T, N...>>,

            traits::extents<nTuple<T, N...>>>
            type;

};
//---------------------------------------------------------------------------------------------------------------------


}  // namespace traits

template<typename T, size_t N> using Vector=nTuple<T, N>;

template<typename T, size_t M, size_t N> using Matrix=nTuple<nTuple<T, N>, M>;

namespace _impl
{

template<typename T, size_t ...> struct make_tensor;

template<typename T> struct make_tensor<T> { typedef T type; };

template<typename T, size_t N0, size_t ...N>
struct make_tensor<T, N0, N...> { typedef nTuple<typename make_tensor<T, N...>::type, N0> type; };

}

template<typename T, size_t ...N> using Tensor=typename _impl::make_tensor<T, N...>::type;

//----------------------------------------------------------------------------------------------------------------------
namespace traits
{

namespace _impl
{
template<typename T, size_t ...M, typename ...Others> void
assigne_nTuple_helper(nTuple<T, M...> &lhs, std::integral_constant<int, 0> const, Others &&... others)
{
}


template<typename T, size_t N, size_t ...M, size_t I, typename T0, typename ...Others> void
assigne_nTuple_helper(nTuple<T, N, M...> &lhs, std::integral_constant<int, I> const, T0 const &a0,
                      Others &&... others)
{
    lhs[N - I] = a0;

    assigne_nTuple_helper(lhs, std::integral_constant<int, I - 1>(), std::forward<Others>(others)...);
}


}//namespace _impl{


template<typename T, size_t ...M, typename ...Others>
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
}//namespace traits

//----------------------------------------------------------------------------------------------------------------------




template<typename T> inline auto determinant(nTuple<T, 3> const &m) DECL_RET_TYPE(((m[0] * m[1] * m[2])))

template<typename T> inline auto determinant(nTuple<T, 4> const &m) DECL_RET_TYPE((m[0] * m[1] * m[2] * m[3]))

template<typename T> inline auto determinant(Matrix<T, 3, 3> const &m)
DECL_RET_TYPE((
                      m[0][0] * m[1][1] * m[2][2] -
                      m[0][2] * m[1][1] * m[2][0] +
                      m[0][1] * m[1][2] * m[2][0] -
                      m[0][1] * m[1][0] * m[2][2] +
                      m[1][0] * m[2][1] * m[0][2] -
                      m[1][2] * m[2][1] * m[0][0]))

template<typename T>
inline auto determinant(Matrix<T, 4, 4> const &m)
DECL_RET_TYPE((
                      m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1] * m[3][0]//
                      - m[0][3] * m[1][1] * m[2][2] * m[3][0] + m[0][1] * m[1][3]//
                                                                * m[2][2] * m[3][0] +
                      m[0][2] * m[1][1] * m[2][3] * m[3][0] -
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
              ))

template<typename T1, size_t ... N1, typename T2, size_t ... N2>
inline auto cross(nTuple<T1, N1...> const &l, nTuple<T2, N2...> const &r)
DECL_RET_TYPE((
                      nTuple<decltype(traits::index(l, 0) * traits::index(r, 0)), 3> {

                              traits::index(l, 1) * traits::index(r, 2) - traits::index(l, 2) * traits::index(r, 1),
                              traits::index(l, 2) * traits::index(r, 0) - traits::index(l, 0) * traits::index(r, 2),
                              traits::index(l, 0) * traits::index(r, 1) - traits::index(l, 1) * traits::index(r, 0)
                      }
              ))
//----------------------------------------------------------------------------------------------------------------------
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
        constexpr nTuple<Expression<_impl::_NAME_,nTuple<T,N...> >> \
        operator _OP_(nTuple<T,N...> const &l)  \
        {return (nTuple<Expression<_impl::_NAME_,nTuple<T,N...> >>(l)) ;}    \


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
        constexpr nTuple<Expression<_impl::_##_NAME_,nTuple<T,N...>>> \
        _NAME_(nTuple<T,N ...> const &r)  \
        {return (nTuple<Expression<_impl::_##_NAME_,nTuple<T,N...>>>(r));}     \


DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA2(nTuple)

/** @}*/
//----------------------------------------------------------------------------------------------------------------------


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
        constexpr nTuple<BooleanExpression<_impl::_NAME_,nTuple<T,N...> >> \
        operator _OP_(nTuple<T,N...> const &l)  \
        {return (nTuple<BooleanExpression<_impl::_NAME_,nTuple<T,N...> >>(l)) ;}    \


DEFINE_EXPRESSOPM_TEMPLATE_BOOLEAN_ALGEBRA2(nTuple)

//----------------------------------------------------------------------------------------------------------------------



template<typename TOP, typename T> T
reduce(TOP const &op, T const &v) { return v; }

template<typename TOP, typename T, size_t N0, size_t ...N>
traits::value_type_t<nTuple<T, N0, N...>>
reduce(TOP const
       &op,
       nTuple<T, N0, N...> const &v
)
{
    static constexpr size_t n = N0;

    traits::value_type_t<nTuple<T, N0, N...> >
            res = reduce(op,
                         traits::index(v, 0));
    if (n > 1)
    {
        for (
                size_t s = 1;
                s < n;
                ++s)
        {
            res = op(res, reduce(op, traits::index(v, s)));
        }
    }
    return
            res;
}

template<typename TOP, typename ...T>
traits::value_type_t<nTuple<Expression<T...> > >
reduce(TOP const
       &op,
       nTuple<Expression<T...>> const &v
)
{
    traits::primary_type_t<nTuple<Expression<T...> > >
            res = v;

    return
            reduce(op, res
            );
}


template<typename TOP, typename ...Args>
void for_each(TOP const &op, index_sequence<>, Args &&... args)
{
    op(std::forward<Args>(args) ...);
}

template<size_t N, size_t ...M, typename TOP, typename ...Args>
void for_each(TOP const &op, index_sequence<N, M...>,
              Args &&... args)
{
    for (size_t s = 0; s < N; ++s)
    {
        for_each(op, index_sequence<M...>(), traits::index(std::forward<Args>(args), s)...);
    }

}

template<typename TR, typename T, size_t ... N>
auto inner_product(nTuple<T, N...> const &l, TR const &r)
DECL_RET_TYPE((reduce(_impl::plus(), l * r)))


inline constexpr double dot(double const &l, double const &r) { return r * l; }

inline constexpr float dot(float const &l, float const &r) { return r * l; }

template<typename TR, typename T, size_t ... N>
auto dot(nTuple<T, N...> const &l, TR const &r) DECL_RET_TYPE((inner_product(l, r)))

template<typename T, size_t ... N>
auto normal(nTuple<T, N...> const &l) DECL_RET_TYPE((std::sqrt(inner_product(l, l))))

template<typename T> auto sp_abs(T const &v) DECL_RET_TYPE((std::abs(v)))

template<typename T, size_t ...N>
auto sp_abs(nTuple<T, N...> const &m) DECL_RET_TYPE((std::sqrt(std::abs(inner_product(m, m)))))

template<typename ... T>
auto sp_abs(nTuple<Expression<T...>> const &m) DECL_RET_TYPE((std::sqrt(std::abs(inner_product(m, m)))))

template<typename T> auto mod(T const &v) DECL_RET_TYPE((sp_abs(v)))

template<typename T, size_t ...N>
auto abs(nTuple<T, N...> const &v) DECL_RET_TYPE((sp_abs(v)))

template<typename T, size_t ...N>
inline auto NProduct(nTuple<T, N...> const &v) DECL_RET_TYPE((reduce(_impl::multiplies(), v)))

template<typename T, size_t ...N>
inline auto NSum(nTuple<T, N...> const &v) DECL_RET_TYPE((reduce(_impl::plus(), v)))

}//namespace simpla

namespace std
{

template<typename T, size_t N, size_t ... M>
void swap(simpla::nTuple<T, N, M...> &l, simpla::nTuple<T, N, M...> &r)
{
    for (size_t s = 0; s < N; ++s)
    {
        swap(simpla::traits::index(l, s), simpla::traits::index(r, s));
    }
}

template<typename T, size_t N, size_t ... M>
void swap(simpla::nTuple<T, N, M...> &l,
          simpla::traits::pod_type_t<simpla::nTuple<T, N, M...>> &r)
{

    for (size_t s = 0; s < N; ++s)
    {
        swap(simpla::traits::index(l, s), simpla::traits::index(r, s));
    }
}

}//namespace std



#endif //SIMPLA_NTUPLELITE_H
