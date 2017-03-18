/** 
 * @file nTupleLite.h
 * @author salmon
 * @date 16-5-27 - 上午8:56
 *  */

#ifndef SIMPLA_NTUPLELITE_H
#define SIMPLA_NTUPLELITE_H

#include <simpla/SIMPLA_config.h>
#include <cstdbool>
#include <cstddef>
#include <type_traits>

#include "simpla/toolbox/macro.h"
#include "simpla/toolbox/mpl.h"
#include "simpla/toolbox/integer_sequence.h"
#include "simpla/toolbox/port_cxx14.h"
#include "simpla/toolbox/type_traits.h"
#include "PrimaryOps.h"
#include "Expression.h"

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
        for (int i = 0; i < N0; ++i) { algebra::assign(op, data_[i], algebra::getValue(rhs, i)); }
    }

    template<typename TR>
    void assign_(std::nullptr_t, TR const &rhs)
    {
        for (int i = 0; i < N0; ++i) { data_[i] = algebra::getValue(rhs, i); }
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
        assign_(reinterpret_cast<algebra::tags::plus *>(nullptr), rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator-=(TR const &rhs)
    {
        assign_(reinterpret_cast<algebra::tags::minus *>(nullptr), rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator*=(TR const &rhs)
    {
        assign_(reinterpret_cast<algebra::tags::multiplies *>(nullptr), rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator/=(TR const &rhs)
    {
        assign_(reinterpret_cast<algebra::tags::divides *>(nullptr), rhs);
        return (*this);
    }

};

template<typename T, size_type N> using Vector=nTuple<T, N>;

template<typename T, size_type M, size_type N> using Matrix=nTuple<nTuple<T, N>, M>;

//namespace traits
//{
//
//
//template<typename>
//struct is_ntuple { static constexpr bool value = false; };
//
//template<typename T, size_type ...N>
//struct is_ntuple<nTuple<T, N...>> { static constexpr bool value = true; };
//
//template<typename T, size_type N>
//struct key_type<nTuple<T, N >> { typedef size_type value_type_info; };
//
//
////----------------------------------------------------------------------------------------------------------------------
//
//template<typename T> struct rank;
//
//
//template<typename T, size_type N> struct rank<nTuple<T, N>> : public int_const<rank<T>::value + 1> {};
//
//
//template<typename TV, size_type N> struct extent<TV, N> : public int_const<0> {};
//
//template<typename TV> struct extent<TV, 0> : public int_const<1> {};
//
//template<typename TV, size_type M, size_type N> struct extent<nTuple<TV, M>, N> : public extent<TV, N - 1> {};
//
//template<typename TV, size_type M> struct extent<nTuple<TV, M>, 0> : public int_const<M> {};
//
//template<typename T, size_type I0> struct size<nTuple<T, I0> > :
//        public std::integral_constant<size_type, I0>
//{
//};
//template<typename T, size_type I0, size_type ...I> struct size<nTuple<T, I0, I...> > :
//        public std::integral_constant<size_type, I0 * size<nTuple<T, I...>>::value>
//{
//};
//
//template<typename ...>
//struct expr_extents_helper;
//template<typename ...>
//struct longest_seq;
//
///**
// * integer sequence of the number of element along all dimensions
// * i.e.
// *
// */
//template<typename>
//struct make_extents { typedef int_sequence<> value_type_info; };
//
//template<typename T, int N>
//struct make_extents<T[N]>
//{
//    typedef traits::seq_concat<int_sequence<N>, typename make_extents<T>::value_type_info> type;
//};
//
//template<typename T, size_type N>
//struct make_extents<nTuple<T, N>>
//{
//    typedef typename traits::seq_concat<int_sequence<N>, typename make_extents<T>::type> value_type_info;
//};
//
//template<typename TOP, typename ... T>
//struct make_extents<algebra::Expression<TOP, T...> >
//{
//    typedef typename expr_extents_helper<algebra::Expression<TOP, T...>>::type value_type_info;
//};
//
//template<size_type ...N>
//struct longest_seq<int_sequence<N...>, int_sequence<> >
//{
//    typedef int_sequence<N...> value_type_info;
//};
//
//template<size_type ...N>
//struct longest_seq<int_sequence<>, int_sequence<N...> >
//{
//    typedef int_sequence<N...> value_type_info;
//};
//
//template<size_type ...N, size_type ...M>
//struct longest_seq<int_sequence<N...>, int_sequence<M...> >
//{
//
//
//    typedef std::conditional_t<(sizeof...(N) < sizeof...(M)), int_sequence<N...>,
//            int_sequence<M...> > value_type_info;
//};
//
//template<typename TOP>
//struct expr_extents_helper<algebra::Expression<TOP >>
//{
//    typedef int_sequence<> value_type_info;
//};
//template<typename TOP, typename First>
//struct expr_extents_helper<algebra::Expression<TOP, First> >
//{
//    typedef typename make_extents<First>::type value_type_info;
//};
//template<typename TOP, typename First, typename ...Others>
//struct expr_extents_helper<algebra::Expression<TOP, First, Others...>>
//{
//    typedef typename longest_seq<typename make_extents<First>::value_type_info,
//            typename make_extents<Others>::type...>::type value_type_info;
//};
//
//template<typename T> using extents=typename _detail::make_extents<T>::value_type_info;
//
//
////----------------------------------------------------------------------------------------------------------------------
//template<typename> struct primary_type;
//
//template<typename> struct pod_type;
//
//namespace _detail
//{
//
//template<typename ...> struct make_pod_array;
//template<typename ...> struct make_primary_nTuple;
//
//template<typename TV, typename TI>
//struct make_pod_array<TV, integer_sequence<TI>>
//{
//    typedef TV value_type_info;
//};
//template<typename TV, typename TI, TI N0, TI ... N>
//struct make_pod_array<TV, integer_sequence<TI, N0, N...>>
//{
//    typedef typename make_pod_array<TV, integer_sequence<TI, N...>>
//    ::value_type_info type[N0];
//};
//
//template<typename TV, typename TI, TI ... N>
//struct make_primary_nTuple<TV, integer_sequence<TI, N...>>
//{
//    typedef nTuple<TV, N...> value_type_info;
//};
//template<typename TV, typename TI>
//struct make_primary_nTuple<TV, integer_sequence<TI>>
//{
//    typedef TV value_type_info;
//};
//
//template<typename ... T> using make_pod_array_t = typename make_pod_array<T...>::value_type_info;
//template<typename ... T> using make_primary_nTuple_t = typename make_primary_nTuple<T...>::value_type_info;
//
//}// namespace _detail
//
//template<typename T, int ...N>
//struct primary_type<nTuple<T, N...>>
//{
//    typedef _detail::make_primary_nTuple_t<
//            algebra::traits::value_type_t<nTuple<T, N...>>, traits::extents<nTuple<T, N...>>>
//            value_type_info;
//};
//
//template<typename T> using ntuple_cast_t=typename primary_type<T>::value_type_info;
//
//template<typename T, size_type ...N>
//struct pod_type<nTuple<T, N...>>
//{
//    typedef _detail::make_pod_array_t<
//            algebra::traits::value_type_t<nTuple<T, N...>>,
//            traits::extents<nTuple<T, N...>>>
//            value_type_info;
//
//};
//---------------------------------------------------------------------------------------------------------------------
//
//}  // namespace traits

//namespace _detail
//{
//
//template<typename T, size_type ...>
//struct make_tensor;
//
//template<typename T>
//struct make_tensor<T> { typedef T value_type_info; };
//
//template<typename T, size_type N0, size_type ...N>
//struct make_tensor<T, N0, N...> { typedef nTuple<typename make_tensor<T, N...>::type, N0> value_type_info; };
//
//}
//
//template<typename T, size_type ...N> using Tensor=typename _detail::make_tensor<T, N...>::value_type_info;
//
////----------------------------------------------------------------------------------------------------------------------
//namespace traits
//{
//
//namespace _detail
//{
//template<typename T, size_type ...M, typename ...Others>
//void
//assigne_nTuple_helper(nTuple<T, M...> &lhs, std::integral_constant<int, 0> const, Others &&... others)
//{
//}
//
//template<typename T, size_type N, size_type ...M, size_type I, typename T0, typename ...Others>
//void
//assigne_nTuple_helper(nTuple<T, N, M...> &lhs, std::integral_constant<int, I> const, T0 const &a0,
//                      Others &&... others)
//{
//    lhs[N - I] = a0;
//
//    assigne_nTuple_helper(lhs, std::integral_constant<int, I - 1>(), std::forward<Others>(others)...);
//}
//
//}//namespace _detail{
//
//
//
//
//template<typename T, size_type ...M, typename ...Others>
//nTuple<T, 1 + sizeof...(Others), M...> make_nTuple(nTuple<T, M...> const &a0, Others &&... others)
//{
//    nTuple<T, 1 + sizeof...(Others), M...> res;
//
//    _detail::assigne_nTuple_helper(res, std::integral_constant<int, 1 + sizeof...(Others)>(), a0,
//                                   std::forward<Others>(others)...);
//
//    return std::Move(res);
//}
//
//template<typename T0, typename ...Others>
//nTuple<T0, 1 + sizeof...(Others)> make_nTuple(T0 const &a0, Others &&... others)
//{
//    nTuple<T0, 1 + sizeof...(Others)> res;
//
//    _detail::assigne_nTuple_helper(res, std::integral_constant<int, 1 + sizeof...(Others)>(), a0,
//                                   std::forward<Others>(others)...);
//
//    return std::Move(res);
//}
//}//namespace traits

} //namespace simpla

//namespace std
//{
//
//
//template<typename T, size_type N>
//void swap(simpla::nTuple<T, N> &l, simpla::nTuple<T, N> &r)
//{
//    for (size_type s = 0; s < N; ++s) { std::swap(l[s], r[s]); }
//}
//
//template<typename T, size_type N, size_type M0, size_type ... M>
//void swap(simpla::nTuple<T, N, M0, M...> &l, simpla::nTuple<T, N, M0, M...> &r)
//{
//    for (size_type s = 0; s < N; ++s) { swap(simpla::traits::GetValue(l, s), simpla::traits::GetValue(r, s)); }
//}
//
//template<typename T, size_type N, size_type ... M>
//void swap(simpla::nTuple<T, N, M...> &l,
//          simpla::traits::pod_type_t<simpla::nTuple<T, N, M...>> &r)
//{
//
//    for (size_type s = 0; s < N; ++s)
//    {
//        swap(simpla::traits::GetValue(l, s), simpla::traits::GetValue(r, s));
//    }
//}
//
//}//namespace std



#endif //SIMPLA_NTUPLELITE_H
