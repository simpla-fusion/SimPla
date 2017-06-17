/**
 * @file integer_sequence.h
 *
 *  Created on: 2014-9-26
 *      Author: salmon
 */

#ifndef CORE_toolbox_INTEGER_SEQUENCE_H_
#define CORE_toolbox_INTEGER_SEQUENCE_H_

#include <simpla/utilities/sp_def.h>
#include <stddef.h>
#include <iostream>
#include "simpla/concept/CheckConcept.h"

namespace simpla {
using namespace concept;
//////////////////////////////////////////////////////////////////////
/// integer_sequence
//////////////////////////////////////////////////////////////////////
//#if __cplusplus >= 201402L
//
//#include <utility>
////template<typename _Tp, _Tp ... _I> using integer_sequence = std::integer_sequence<_Tp, _I...>;
////template<int ... _I> using index_sequence=std::index_sequence<_I...>;
// using std::integer_sequence;
// using std::int_sequence;
// using std::make_int_sequence;
// using std::make_integer_sequence;
// using std::int_sequence_for;
//#else
/**
 *  alt. of std::integer_sequence ( C++14)
 *  @quto http://en.cppreference.com/w/cpp/utilities/integer_sequence
 *  The class template  integer_sequence represents a
 *  compile-time sequence of integers. When used as an argument
 *   to a function template, the parameter pack Ints can be deduced
 *   and used in pack expansion.
 *
 *
 */
//**************************************************************************
// from standard library
// Stores a tuple of indices.  Used by tuple and pair, and by bind() to
// extract the elements in a tuple.
/// Class template integer_sequence

template <typename _Tp, _Tp... _Idx>
struct integer_sequence {
    typedef _Tp value_type;
    static constexpr int size() { return sizeof...(_Idx); }

    //    template <_Tp... _J>
    //    integer_sequence<_Tp, _Idx..., _J...> operator,(integer_sequence<_Tp, _J...>){};
};

namespace _impl {
template <int... _Indexes>
struct _Index_tuple {
    typedef _Index_tuple<_Indexes..., sizeof...(_Indexes)> __next;
};

// Builds an _Index_tuple<0, 1, 2, ..., _Num-1>.
template <int _Num>
struct _Build_index_tuple {
    typedef typename _Build_index_tuple<_Num - 1>::__type::__next __type;
};

template <>
struct _Build_index_tuple<0> {
    typedef _Index_tuple<> __type;
};

template <typename _Tp, _Tp _Num, typename _ISeq = typename _Build_index_tuple<_Num>::__type>
struct _Make_integer_sequence;

template <typename _Tp, _Tp _Num, int... _Idx>
struct _Make_integer_sequence<_Tp, _Num, _Index_tuple<_Idx...>> {
    static_assert(_Num >= 0, "Cannot make integer sequence of negative length");

    typedef integer_sequence<_Tp, static_cast<_Tp>(_Idx)...> __type;
};
}  // namespace _impl

/// Alias template make_integer_sequence
template <typename _Tp, _Tp _Num>
using make_integer_sequence = typename _impl::_Make_integer_sequence<_Tp, _Num>::__type;

template <int... _Idx>
using index_sequence = integer_sequence<int, _Idx...>;

/// Alias template index_sequence
template <int... _Idx>
using int_sequence = integer_sequence<int, _Idx...>;

/// Alias template make_index_sequence
template <int _Num>
using make_int_sequence = make_integer_sequence<int, _Num>;

template <typename... _Types>
using int_sequence_for = make_int_sequence<sizeof...(_Types)>;

/// Alias template index_sequence_for
template <typename... _Types>
using index_sequence_for = make_int_sequence<sizeof...(_Types)>;
//**************************************************************************

//#endif
namespace tags {
template <int V0, int V1, int V2>
using VERSION = int_sequence<V0, V1, V2>;
}

namespace traits {

template <typename T, typename TI>
auto index(T &v, integer_sequence<TI>, ENABLE_IF((is_indexable<T, TI>::value))) AUTO_RETURN((v));

template <typename T, typename TI, TI M, TI... N>
auto index(T &v, integer_sequence<TI, M, N...>, ENABLE_IF((is_indexable<T, TI>::value)))
    AUTO_RETURN((index(v[M], integer_sequence<TI, N...>())));

//----------------------------------------------------------------------------------------------------------------------

template <typename>
struct seq_value;
template <typename _Tp, _Tp... N>
struct seq_value<integer_sequence<_Tp, N...>> {
    static constexpr _Tp value[] = {N...};
};
template <typename _Tp, _Tp... N>
constexpr _Tp seq_value<integer_sequence<_Tp, N...>>::value[];

//----------------------------------------------------------------------------------------------------------------------

template <int N, typename...>
struct seq_get;

template <int N, typename Tp, Tp M, Tp... I>
struct seq_get<N, integer_sequence<Tp, M, I...>> {
    static constexpr Tp value = seq_get<N - 1, integer_sequence<Tp, I...>>::value;
};

template <typename Tp, Tp M, Tp... I>
struct seq_get<0, integer_sequence<Tp, M, I...>> {
    static constexpr Tp value = M;
};

template <typename Tp>
struct seq_get<0, integer_sequence<Tp>> {
    static constexpr Tp value = 0;
};
template <int N, typename _Tp, _Tp... I>
_Tp get(integer_sequence<_Tp, I...>) {
    return seq_get<N, integer_sequence<_Tp, I...>>::value;
};
// template<typename ...> class longer_integer_sequence;
//
// template<typename T, T ... N>
// struct longer_integer_sequence<integer_sequence<T, N ...> >
//{
//	typedef integer_sequence<T, N ...> value_type_info;
//};
// template<typename T, T ... N, T ...M>
// struct longer_integer_sequence<integer_sequence<T, N ...>,
//		integer_sequence<T, M...>>
//{
//	typedef integer_sequence<T, mpl::max<T,N,M>::entity ...> value_type_info;
//};
//
////template<typename T, T ... N1, T ... N2, typename ...Others>
////struct longer_integer_sequence<integer_sequence<T, N1...>,
////		integer_sequence<T, N2...>, Others ...>
////{
////	typedef typename std::conditional<(sizeof...(N1) > sizeof...(N2)),
////			typename longer_integer_sequence<integer_sequence<T, N1...>,
////					Others...>::type,
////			typename longer_integer_sequence<integer_sequence<T, N2...>,
////					Others...>::type>::type type;
////
////};
//----------------------------------------------------------------------------------------------------------------------

namespace _impl {
// TODO need implement max_integer_sequence, min_integer_sequence
template <int...>
struct _seq_for;

template <int M>
struct _seq_for<M> {
    template <typename TOP, typename... Args>
    static inline void eval(TOP const &op, Args &&... args) {
        op(access(std::forward<Args>(args), M - 1)...);
        _seq_for<M - 1>::eval(op, std::forward<Args>(args)...);
    }
};

template <>
struct _seq_for<0> {
    template <typename TOP, typename... Args>
    static inline void eval(TOP const &op, Args &&... args) {}
};

template <int M, int... N>
struct _seq_for<M, N...> {
    template <typename TOP, typename... Args>
    static inline void eval(TOP const &op, Args &&... args) {
        eval(op, integer_sequence<int>(), std::forward<Args>(args)...);
    }

    template <typename TOP, int... L, typename... Args>
    static inline void eval(TOP const &op, integer_sequence<int, L...>, Args &&... args) {
        _seq_for<N...>::eval(op, integer_sequence<int, L..., M>(), std::forward<Args>(args)...);

        _seq_for<M - 1, N...>::eval(op, integer_sequence<int, L...>(), std::forward<Args>(args)...);
    }
};
}

// namespace _impl{
template <int N, typename... Args>
void seq_for(int_sequence<N>, Args &&... args) {
    _impl::_seq_for<N>::eval(std::forward<Args>(args)...);
}

template <int... N, typename... Args>
void seq_for(int_sequence<N...>, Args &&... args) {
    _impl::_seq_for<N...>::eval(std::forward<Args>(args)...);
}
//----------------------------------------------------------------------------------------------------------------------

namespace _impl {
template <int...>
struct _seq_reduce;

template <int M, int... N>
struct _seq_reduce<M, N...> {
    template <typename Reduction, int... L, typename... Args>
    static inline auto eval(Reduction const &reduction, integer_sequence<int, L...>, Args &&... args)
        AUTO_RETURN((reduction(
            _seq_reduce<N...>::eval(reduction, integer_sequence<int, L..., M>(), std::forward<Args>(args)...),
            _seq_reduce<M - 1, N...>::eval(reduction, integer_sequence<int, L...>(), std::forward<Args>(args)...))));

    template <typename Reduction, typename... Args>
    static inline auto eval(Reduction const &reduction, Args &&... args)
        AUTO_RETURN((eval(reduction, integer_sequence<int>(), std::forward<Args>(args)...)));
};

template <int... N>
struct _seq_reduce<1, N...> {
    template <typename Reduction, int... L, typename... Args>
    static inline auto eval(Reduction const &reduction, integer_sequence<int, L...>, Args &&... args)
        AUTO_RETURN((_seq_reduce<N...>::eval(reduction, integer_sequence<int, L..., 1>(),
                                             std::forward<Args>(args)...)));
};

template <>
struct _seq_reduce<> {
    template <typename Reduction, int... L, typename Args>
    static inline auto eval(Reduction const &, integer_sequence<int, L...>, Args const &args)
        AUTO_RETURN((access(args, integer_sequence<int, (L - 1)...>())));
};
}  // namespace _impl {

// namespace _impl
template <int... N, typename TOP, typename... Args>
auto seq_reduce(integer_sequence<int, N...>, TOP const &op, Args &&... args)
    AUTO_RETURN(((_impl::_seq_reduce<N...>::eval(op, std::forward<Args>(args)...))));

//----------------------------------------------------------------------------------------------------------------------

template <int... N, typename TOP>
void seq_for_each(int_sequence<N...>, TOP const &op) {
    int ndims = sizeof...(N);
    int dims[] = {N...};
    int idx[ndims];

    for (int i = 0; i < ndims; ++i) { idx[i] = 0; }

    while (1) {
        op(idx);

        ++idx[ndims - 1];
        for (int rank = ndims - 1; rank > 0; --rank) {
            if (idx[rank] >= dims[rank]) {
                idx[rank] = 0;
                ++idx[rank - 1];
            }
        }
        if (idx[0] >= dims[0]) { break; }
    }
}

namespace _impl {

template <typename _Tp, _Tp I0, _Tp... I>
struct _seq_max {
    static constexpr _Tp value = _seq_max<_Tp, I0, _seq_max<_Tp, I...>::value>::value;
};
template <typename _Tp, _Tp I0>
struct _seq_max<_Tp, I0> {
    static constexpr _Tp value = I0;
};
template <typename _Tp, _Tp I0, _Tp I1>
struct _seq_max<_Tp, I0, I1> {
    static constexpr _Tp value = (I0 > I1) ? I0 : I1;
};

template <typename _Tp, _Tp I0, _Tp... I>
struct _seq_min {
    static constexpr _Tp value = _seq_min<_Tp, I0, _seq_min<_Tp, I...>::value>::value;
};
template <typename _Tp, _Tp I0>
struct _seq_min<_Tp, I0> {
    static constexpr _Tp value = I0;
};
template <typename _Tp, _Tp I0, _Tp I1>
struct _seq_min<_Tp, I0, I1> {
    static constexpr _Tp value = (I0 < I1) ? I0 : I1;
};

}  // namespace _impl

template <typename Tp>
struct seq_max : public int_sequence<> {};

template <typename _Tp, _Tp... I>
struct seq_max<integer_sequence<_Tp, I...>> : public std::integral_constant<_Tp, _impl::_seq_max<_Tp, I...>::value> {};

template <typename Tp>
struct seq_min : public int_sequence<> {};

template <typename _Tp, _Tp... I>
struct seq_min<integer_sequence<_Tp, I...>> : public std::integral_constant<_Tp, _impl::_seq_min<_Tp, I...>::value> {};

//----------------------------------------------------------------------------------------------------------------------

namespace _impl {

/**
 *  cat two tuple/integer_sequence
 */
template <typename...>
struct seq_concat_helper;

template <typename _Tp, _Tp... _M>
struct seq_concat_helper<integer_sequence<_Tp, _M...>> {
    typedef integer_sequence<_Tp, _M...> type;
};

template <typename _Tp, _Tp... _M, _Tp... _N>
struct seq_concat_helper<integer_sequence<_Tp, _M...>, integer_sequence<_Tp, _N...>> {
    typedef integer_sequence<_Tp, _M..., _N...> type;
};

template <typename _Tp, _Tp... _M>
struct seq_concat_helper<integer_sequence<_Tp, _M...>, integer_sequence<_Tp>> {
    typedef integer_sequence<_Tp, _M...> type;
};
template <typename _Tp, typename... Others>
struct seq_concat_helper<_Tp, Others...> {
    typedef typename seq_concat_helper<_Tp, typename seq_concat_helper<Others...>::type>::type type;
};

}  // namespace _impl{

template <typename... T>
using seq_concat = typename _impl::seq_concat_helper<T...>::type;

//----------------------------------------------------------------------------------------------------------------------

template <typename TInts, TInts... N, typename TA>
std::ostream &seq_print(integer_sequence<TInts, N...>, std::ostream &os, TA const &d) {
    int ndims = sizeof...(N);
    TInts dims[] = {N...};
    TInts idx[ndims];

    for (int i = 0; i < ndims; ++i) { idx[i] = 0; }

    while (1) {
        os << access(d, idx) << ", ";

        ++idx[ndims - 1];

        for (int rank = ndims - 1; rank > 0; --rank) {
            if (idx[rank] >= dims[rank]) {
                idx[rank] = 0;
                ++(idx[rank - 1]);

                if (rank == ndims - 1) { os << "\n"; }
            }
        }
        if (idx[0] >= dims[0]) { break; }
    }
    return os;
}

}  // namespace traits

template <typename _Tp, _Tp First, _Tp Second, _Tp... Others>
std::ostream &operator<<(std::ostream &os, integer_sequence<_Tp, First, Second, Others...> const &) {
    os << First << " , " << integer_sequence<_Tp, Second, Others...>();
    return os;
}

template <typename _Tp, _Tp First>
std::ostream &operator<<(std::ostream &os, integer_sequence<_Tp, First> const &) {
    os << First;
    return os;
}

template <typename _Tp>
std::ostream &operator<<(std::ostream &os, integer_sequence<_Tp> const &) {
    return os;
}

template <int I>
using int_const = std::integral_constant<int, I>;

template <typename _Tp, _Tp I>
using integral_constant = std::integral_constant<_Tp, I>;
template <int... I>
using int_sequence = integer_sequence<int, I...>;
static const integer_sequence<int, 0> _0{};
static const integer_sequence<int, 1> _1{};
static const integer_sequence<int, 2> _2{};
static const integer_sequence<int, 3> _3{};
static const integer_sequence<int, 4> _4{};
static const integer_sequence<int, 5> _5{};
static const integer_sequence<int, 6> _6{};
static const integer_sequence<int, 7> _7{};
static const integer_sequence<int, 8> _8{};
static const integer_sequence<int, 9> _9{};

template <typename _T1>
auto operator-(integer_sequence<_T1>) AUTO_RETURN((integer_sequence<_T1>()));

template <typename _T1, _T1 I0, _T1... I>
auto operator-(integer_sequence<_T1, I0, I...>)
    AUTO_RETURN((integer_sequence<_T1, -I0>(), (-integer_sequence<_T1, I...>())));

template <typename _T1, _T1... I, typename _T2>
auto operator+(integer_sequence<_T1, I...>, integer_sequence<_T2>) AUTO_RETURN((integer_sequence<_T1, I...>()));

template <typename _T1, typename _T2, _T2... J>
auto operator+(integer_sequence<_T1>, integer_sequence<_T2, J...>) AUTO_RETURN((integer_sequence<_T2, J...>()));

template <typename _T1>
auto operator+(integer_sequence<_T1>, integer_sequence<_T1>) AUTO_RETURN((integer_sequence<_T1>()));

template <typename _T1, _T1 I0, _T1... I, typename _T2, _T2 J0, _T2... J>
auto operator+(integer_sequence<_T1, I0, I...>, integer_sequence<_T2, J0, J...>)
    AUTO_RETURN((integral_constant<_T1, (I0 + J0)>(), (integer_sequence<_T1, I...>() + integer_sequence<_T2, J...>())));

template <typename _T1, _T1... I, typename _T2>
auto operator-(integer_sequence<_T1, I...>, integer_sequence<_T2>) AUTO_RETURN((integer_sequence<_T1, I...>()));

template <typename _T1, typename _T2, _T2... J>
auto operator-(integer_sequence<_T1>, integer_sequence<_T2, J...>) AUTO_RETURN((-integer_sequence<_T2, J...>()));

template <typename _T1>
auto operator-(integer_sequence<_T1>, integer_sequence<_T1>) AUTO_RETURN((integer_sequence<_T1>()));

template <typename _T1, _T1 I0, _T1... I, typename _T2, _T2 J0, _T2... J>
auto operator-(integer_sequence<_T1, I0, I...>, integer_sequence<_T2, J0, J...>)
    AUTO_RETURN((integral_constant<_T1, (I0 - J0)>(), (integer_sequence<_T1, I...>() - integer_sequence<_T2, J...>())));

template <typename _T1, _T1... I, typename _T2, _T2 M>
auto operator*(integer_sequence<_T1, I...>, integral_constant<_T2, M>)AUTO_RETURN((integer_sequence<_T1, I * M...>()));

template <typename _T1, _T1... I, typename _T2, _T2... J>
auto operator,(integer_sequence<_T1, I...>, integer_sequence<_T2, J...>)
    AUTO_RETURN((integer_sequence<_T1, I..., J...>()));

template <typename _Tp, _Tp... N>
struct seq_max;
template <typename _Tp, _Tp N0>
struct seq_max<_Tp, N0> : public integral_constant<_Tp, N0> {};

template <typename _Tp, _Tp N0, _Tp N1>
struct seq_max<_Tp, N0, N1> : public integral_constant<_Tp, (N0 > N1 ? N0 : N1)> {};

template <typename _Tp, _Tp N0, _Tp... N>
struct seq_max<_Tp, N0, N...> : public integral_constant<_Tp, seq_max<_Tp, N0, seq_max<_Tp, N...>::value>::value> {};

template <typename _Tp, _Tp... N>
struct seq_min;

template <typename _Tp, _Tp N0, _Tp... N>
struct seq_min<_Tp, N0, N...> : public integral_constant<_Tp, seq_min<_Tp, N0, seq_min<_Tp, N...>::value>::value> {};

template <typename _Tp, _Tp N0>
struct seq_min<_Tp, N0> : public integral_constant<_Tp, N0> {};

template <typename _Tp, _Tp N0, _Tp N1>
struct seq_min<_Tp, N0, N1> : public integral_constant<_Tp, (N0 < N1 ? N0 : N1)> {};

}  // namespace simpla
#endif /* CORE_toolbox_INTEGER_SEQUENCE_H_ */
