/**
 * @file mpl.h
 *
 *  Created on: 2015-6-12
 *      Author: salmon
 */

#ifndef CORE_GTL_MPL_H_
#define CORE_GTL_MPL_H_

#include <tuple>

namespace simpla
{

namespace mpl
{
template<size_t N, typename _TP, _TP ...I> struct unpack_int_seq;
template<size_t N, typename _TP, _TP ...I>
using unpack_int_seq_t=typename unpack_int_seq<N, _TP, I...>::type;

template<typename _Tp, _Tp I0, _Tp ...I>
struct unpack_int_seq<0, _Tp, I0, I...> : public std::integral_constant<_Tp, I0>
{

};
template<size_t N, typename _Tp, _Tp I0, _Tp ...I>
struct unpack_int_seq<N, _Tp, I0, I...> : public std::integral_constant<_Tp,
        unpack_int_seq<N - 1, _Tp, I...>::value>
{
};

template<size_t N, typename _Tp>
struct unpack_int_seq<N, _Tp> : public std::integral_constant<_Tp, 0>
{
};

template<unsigned int, typename ...> struct unpack_type_seq;
template<unsigned int N, typename ...T>
using unpack_type_seq_t=typename unpack_type_seq<N, T...>::type;

template<typename T0, typename ...Others>
struct unpack_type_seq<0, T0, Others...>
{
    typedef T0 type;
};
template<unsigned int N>
struct unpack_type_seq<N>
{
    typedef void type;
};
template<unsigned int N, typename T0, typename ...Others>
struct unpack_type_seq<N, T0, Others...>
{
    typedef typename unpack_type_seq<N - 1, Others...>::type type;
};

template<typename, typename ...> struct find_type_in_list;
template<typename T, typename ...Others>
using find_type_in_list_t=typename find_type_in_list<T, Others...>::type;

template<typename T>
struct find_type_in_list<T>
{
    static constexpr bool value = false;
};
template<typename T, typename U>
struct find_type_in_list<T, U>
{
    static constexpr bool value = std::is_same<T, U>::value;
};
template<typename T, typename U, typename ...Others>
struct find_type_in_list<T, U, Others...>
{
    static constexpr bool value = find_type_in_list<T, U>::value
                                  || find_type_in_list<T, Others...>::value;
};

template<typename _Tp, _Tp ... Others> struct max;

template<typename _Tp, _Tp first>
struct max<_Tp, first> : std::integral_constant<_Tp, first>
{
};

template<typename _Tp, _Tp first, _Tp second>
struct max<_Tp, first, second> : std::integral_constant<_Tp,
        (first > second) ? first : second>
{
};
template<typename _Tp, _Tp first, _Tp ... Others>
struct max<_Tp, first, Others...> : std::integral_constant<_Tp,
        max<_Tp, first, max<_Tp, Others...>::value>::value>
{
};

template<typename _Tp, _Tp ... Others> struct min;

template<typename _Tp, _Tp first>
struct min<_Tp, first> : std::integral_constant<_Tp, first>
{
};
template<typename _Tp, _Tp first, _Tp second>
struct min<_Tp, first, second> : std::integral_constant<_Tp,
        (first > second) ? first : second>
{
};
template<typename _Tp, _Tp first, _Tp ... Others>
struct min<_Tp, first, Others...> : std::integral_constant<_Tp,
        min<_Tp, first, min<_Tp, Others...>::value>::value>
{
};
template<bool...> struct logical_or;
template<bool first>
struct logical_or<first> : public std::integral_constant<bool, first>::type
{
};
template<bool first, bool second>
struct logical_or<first, second> : public std::integral_constant<bool,
        first || second>::type
{

};
template<bool first, bool ... args>
struct logical_or<first, args...> : public std::integral_constant<bool,
        logical_or<first, logical_or<args...>::value>::value>::type
{
};

template<typename ...T> struct cat_tuple;
template<typename ...T> using cat_tuple_t=typename cat_tuple<T...>::type;

template<typename T0>
struct cat_tuple<T0>
{
    typedef std::tuple<T0> type;
};
template<typename ... T>
struct cat_tuple<std::tuple<T...>>
{
    typedef std::tuple<T...> type;
};
template<>
struct cat_tuple<>
{
    typedef std::tuple<> type;
};

template<typename T0, typename ...T>
struct cat_tuple<T0, T...>
{
    typedef cat_tuple_t<cat_tuple_t<T0>, cat_tuple_t<T...>> type;
};

template<typename ...T0, typename ...T1>
struct cat_tuple<std::tuple<T0...>, std::tuple<T1...>>
{
    typedef std::tuple<T0..., T1...> type;
};
template<size_t I, typename ...T>
struct split_tuple;
template<size_t I, typename ...T> using split_tuple_t=typename split_tuple<I, T...>::type;

template<size_t I>
struct split_tuple<I>
{
    typedef std::tuple<> previous;
    typedef std::tuple<> following;
};
template<typename T0, typename ...T>
struct split_tuple<0, T0, T...>
{
    typedef std::tuple<> previous;
    typedef std::tuple<T0, T...> following;
};
template<size_t I, typename T0, typename ...T>
struct split_tuple<I, T0, T...>
{
    typedef cat_tuple_t<T0, typename split_tuple<I - 1, T...>::previous> previous;
    typedef typename split_tuple<I - 1, T...>::following following;
};

template<template<typename ...> class H, typename ...P>
struct assamble_tuple
{
    typedef H<P...> type;
};

template<template<typename ...> class H, typename ...P>
struct assamble_tuple<H, std::tuple<P...>>
{
    typedef typename assamble_tuple<H, P...>::type type;
};
template<template<typename ...> class H, typename ...P>
using assamble_tuple_t=typename assamble_tuple<H, P...>::type;

template<size_t I, typename U, typename T> struct replace_tuple;

template<size_t I, typename U>
struct replace_tuple<I, U, std::nullptr_t>
{
    typedef U type;
};

template<size_t I, typename U, typename ...T>
struct replace_tuple<I, U, std::tuple<T...> >
{
    typedef cat_tuple_t<typename split_tuple<I, T...>::previous, U,
            typename split_tuple<I + 1, T...>::following> type;
};

template<size_t I, typename U, typename ...T, template<typename ...> class H>
struct replace_tuple<I, U, H<T...> >
{
    typedef assamble_tuple_t<H, cat_tuple_t<typename split_tuple<I, T...>::previous, U,
            typename split_tuple<I + 1, T...>::following> > type;
};

template<size_t I, typename U, typename T> using replace_tuple_t=
typename replace_tuple<I, U, T>::type;

}
// namespace mpl

}// namespace simpla

#endif /* CORE_GTL_MPL_H_ */
