/**
 * @file integer_sequence.h
 *
 *  Created on: 2014-9-26
 *      Author: salmon
 */

#ifndef CORE_toolbox_INTEGER_SEQUENCE_H_
#define CORE_toolbox_INTEGER_SEQUENCE_H_

#include <stddef.h>
#include <iostream>
#include "CheckConcept.h"
#include "port_cxx14.h"
#include "macro.h"

namespace simpla
{

//////////////////////////////////////////////////////////////////////
/// integer_sequence
//////////////////////////////////////////////////////////////////////
#if __cplusplus >= 201402L

#include <utility>
//template<typename _Tp, _Tp ... _I> using integer_sequence = std::integer_sequence<_Tp, _I...>;
//template<int ... _I> using index_sequence=std::index_sequence<_I...>;
using std::integer_sequence;
using std::index_sequence;
using std::make_index_sequence;
using std::make_integer_sequence;
using std::index_sequence_for;
#else
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
//from standard library
// Stores a tuple of indices.  Used by tuple and pair, and by bind() to
// extract the elements in a tuple.
/// Class template integer_sequence

template<typename _Tp, _Tp... _Idx>
struct integer_sequence
{
    typedef _Tp value_type;

    static constexpr size_type size() { return sizeof...(_Idx); }
};
namespace _impl
{
template<size_t... _Indexes>
struct _Index_tuple
{
    typedef _Index_tuple<_Indexes..., sizeof...(_Indexes)> __next;
};

// Builds an _Index_tuple<0, 1, 2, ..., _Num-1>.
template<size_type _Num>
struct _Build_index_tuple
{
    typedef typename _Build_index_tuple<_Num - 1>::__type::__next __type;
};

template<>
struct _Build_index_tuple<0>
{
    typedef _Index_tuple<> __type;
};

template<typename _Tp, _Tp _Num,
        typename _ISeq = typename _Build_index_tuple<_Num>::__type>
struct _Make_integer_sequence;

template<typename _Tp, _Tp _Num, size_t... _Idx>
struct _Make_integer_sequence<_Tp, _Num, _Index_tuple<_Idx...>>
{
    static_assert(_Num >= 0, "Cannot make integer sequence of negative length");

    typedef integer_sequence<_Tp, static_cast<_Tp>(_Idx)...> __type;
};
}//namespace _impl

/// Alias template make_integer_sequence
template<typename _Tp, _Tp _Num>
using make_integer_sequence= typename _impl::_Make_integer_sequence<_Tp, _Num>::__type;

/// Alias template index_sequence
template<size_type ... _Idx>
using index_sequence = integer_sequence<size_type, _Idx...>;

/// Alias template make_index_sequence
template<size_type _Num>
using make_index_sequence = make_integer_sequence<size_type, _Num>;

/// Alias template index_sequence_for
template<typename... _Types>
using index_sequence_for = make_index_sequence<sizeof...(_Types)>;
//**************************************************************************


#endif
namespace tags { template<size_type V0, size_type V1, size_type V2> using VERSION = integer_sequence<size_t, V0, V1, V2>; }

namespace traits
{

template<typename T, typename TI>
auto index(T &v, integer_sequence<TI>, ENABLE_IF((is_indexable<T, TI>::value))) DECL_RET_TYPE(v)

template<typename T, typename TI, TI M, TI ...N>
auto index(T &v, integer_sequence<TI, M, N...>, ENABLE_IF((is_indexable<T, TI>::value))) DECL_RET_TYPE(
        (index(v[M], integer_sequence<TI, N...>())))
//----------------------------------------------------------------------------------------------------------------------

template<typename> struct seq_value;
template<typename _Tp, _Tp ...N>
struct seq_value<integer_sequence<_Tp, N...> > { static constexpr _Tp value[] = {N...}; };
template<typename _Tp, _Tp ...N>
constexpr _Tp seq_value<integer_sequence<_Tp, N...>>::value[];

//----------------------------------------------------------------------------------------------------------------------

template<size_type N, typename ...> struct seq_get;

template<size_type N, typename Tp, Tp M, Tp ...I>
struct seq_get<N, integer_sequence<Tp, M, I ...> >
{
    static constexpr Tp value =
            seq_get<N - 1, integer_sequence<Tp, I ...> >::value;
};

template<typename Tp, Tp M, Tp ...I>
struct seq_get<0, integer_sequence<Tp, M, I ...> >
{
    static constexpr Tp value = M;
};

template<typename Tp>
struct seq_get<0, integer_sequence<Tp> >
{
    static constexpr Tp value = 0;
};

//template<typename ...> class longer_integer_sequence;
//
//template<typename T, T ... N>
//struct longer_integer_sequence<integer_sequence<T, N ...> >
//{
//	typedef integer_sequence<T, N ...> type;
//};
//template<typename T, T ... N, T ...M>
//struct longer_integer_sequence<integer_sequence<T, N ...>,
//		integer_sequence<T, M...>>
//{
//	typedef integer_sequence<T, mpl::max<T,N,M>::entity ...> type;
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

namespace _impl
{
//TODO need implement max_integer_sequence, min_integer_sequence
template<size_t...> struct _seq_for;

template<size_type M>
struct _seq_for<M>
{

    template<typename TOP, typename ...Args>
    static inline void eval(TOP const &op, Args &&... args)
    {
        op(access(std::forward<Args>(args), M - 1)...);
        _seq_for<M - 1>::eval(op, std::forward<Args>(args)...);
    }

};

template<>
struct _seq_for<0>
{
    template<typename TOP, typename ...Args> static inline void eval(TOP const &op, Args &&... args) {}
};

template<size_type M, size_type ...N>
struct _seq_for<M, N...>
{

    template<typename TOP, typename ...Args>
    static inline void eval(TOP const &op, Args &&... args)
    {
        eval(op, integer_sequence<size_t>(), std::forward<Args>(args)...);
    }

    template<typename TOP, size_type ...L, typename ...Args>
    static inline void eval(TOP const &op, integer_sequence<size_t, L...>,
                            Args &&... args)
    {
        _seq_for<N...>::eval(op, integer_sequence<size_t, L..., M>(),
                             std::forward<Args>(args)...);

        _seq_for<M - 1, N...>::eval(op, integer_sequence<size_t, L...>(),
                                    std::forward<Args>(args)...);
    }

};
}

//namespace _impl{
template<size_type N, typename ...Args>
void seq_for(index_sequence<N>, Args &&... args)
{
    _impl::_seq_for<N>::eval(std::forward<Args>(args) ...);
}

template<size_type ... N, typename ...Args>
void seq_for(index_sequence<N...>, Args &&... args)
{
    _impl::_seq_for<N...>::eval(std::forward<Args>(args) ...);
}
//----------------------------------------------------------------------------------------------------------------------

namespace _impl
{
template<size_t...> struct _seq_reduce;

template<size_type M, size_type ...N>
struct _seq_reduce<M, N...>
{

    template<typename Reduction, size_type ...L, typename ... Args>
    static inline auto eval(Reduction const &reduction, integer_sequence<size_t, L...>, Args &&... args)
    DECL_RET_TYPE((reduction(
            _seq_reduce<N...>::eval(reduction, integer_sequence<size_t, L..., M>(), std::forward<Args>(args)...),
            _seq_reduce<M - 1, N...>::eval(reduction, integer_sequence<size_t, L...>(), std::forward<Args>(args)...)

    )))

    template<typename Reduction, typename ...Args>
    static inline auto eval(Reduction const &reduction, Args &&... args)

    DECL_RET_TYPE (eval(reduction, integer_sequence<size_t>(), std::forward<Args>(args)...))

};

template<size_type ...N>
struct _seq_reduce<1, N...>
{

    template<typename Reduction, size_type ...L, typename ...Args>
    static inline auto eval(Reduction const &reduction, integer_sequence<size_t, L...>, Args &&... args)
    DECL_RET_TYPE (
            (_seq_reduce<N...>::eval(reduction, integer_sequence<size_t, L..., 1>(), std::forward<Args>(args)...)))

};

template<>
struct _seq_reduce<>
{
    template<typename Reduction, size_type ...L, typename Args>
    static inline auto eval(Reduction const &, integer_sequence<size_t, L...>, Args const &args)

    DECL_RET_TYPE ((access(args, integer_sequence<size_t, (L - 1)...>())))


};
}

//namespace _impl
template<size_type ... N, typename TOP, typename ...Args>
auto seq_reduce(integer_sequence<size_t, N...>, TOP const &op, Args &&... args)
DECL_RET_TYPE((_impl::_seq_reduce<N...>::eval(op, std::forward<Args>(args)...)))
//----------------------------------------------------------------------------------------------------------------------

template<typename TInts, TInts ...N, typename TOP>
void seq_for_each(integer_sequence<TInts, N...>, TOP const &op)
{
    size_type ndims = sizeof...(N);
    TInts dims[] = {N...};
    TInts idx[ndims];

    for (int i = 0; i < ndims; ++i)
    {
        idx[i] = 0;
    }

    while (1)
    {

        op(idx);

        ++idx[ndims - 1];
        for (int rank = ndims - 1; rank > 0; --rank)
        {
            if (idx[rank] >= dims[rank])
            {
                idx[rank] = 0;
                ++idx[rank - 1];
            }
        }
        if (idx[0] >= dims[0])
        {
            break;
        }
    }

}

namespace _impl
{


template<typename _Tp, _Tp I0, _Tp ...I>
struct _seq_max
{
    static constexpr _Tp value = _seq_max<_Tp, I0, _seq_max<_Tp, I...>::value>::value;
};
template<typename _Tp, _Tp I0>
struct _seq_max<_Tp, I0>
{
    static constexpr _Tp value = I0;
};
template<typename _Tp, _Tp I0, _Tp I1>
struct _seq_max<_Tp, I0, I1>
{
    static constexpr _Tp value = (I0 > I1) ? I0 : I1;
};


template<typename _Tp, _Tp I0, _Tp ...I>
struct _seq_min
{
    static constexpr _Tp value = _seq_min<_Tp, I0, _seq_min<_Tp, I...>::value>::value;
};
template<typename _Tp, _Tp I0>
struct _seq_min<_Tp, I0>
{
    static constexpr _Tp value = I0;
};
template<typename _Tp, _Tp I0, _Tp I1>
struct _seq_min<_Tp, I0, I1>
{
    static constexpr _Tp value = (I0 < I1) ? I0 : I1;
};

}//namespace _impl

template<typename Tp> struct seq_max : public index_sequence<> {};

template<typename _Tp, _Tp ...I>
struct seq_max<integer_sequence<_Tp, I...>> : public std::integral_constant<_Tp, _impl::_seq_max<_Tp, I...>::value> {};


template<typename Tp> struct seq_min : public index_sequence<> {};

template<typename _Tp, _Tp ...I>
struct seq_min<integer_sequence<_Tp, I...>> : public std::integral_constant<_Tp, _impl::_seq_min<_Tp, I...>::value> {};

//----------------------------------------------------------------------------------------------------------------------


namespace _impl
{

/**
 *  cat two tuple/integer_sequence
 */
template<typename ...> struct seq_concat_helper;

template<typename _Tp, _Tp ... _M>
struct seq_concat_helper<integer_sequence<_Tp, _M...>>
{
    typedef integer_sequence<_Tp, _M...> type;
};


template<typename _Tp, _Tp ... _M, _Tp ... _N>
struct seq_concat_helper<integer_sequence<_Tp, _M...>, integer_sequence<_Tp, _N...> >
{
    typedef integer_sequence<_Tp, _M..., _N...> type;
};

template<typename _Tp, _Tp ... _M>
struct seq_concat_helper<integer_sequence<_Tp, _M...>, integer_sequence<_Tp> >
{
    typedef integer_sequence<_Tp, _M...> type;
};
template<typename _Tp, typename ...Others>
struct seq_concat_helper<_Tp, Others...>
{
    typedef typename seq_concat_helper<_Tp, typename seq_concat_helper<Others...>::type>::type type;
};


}//namespace _impl{

template<typename ...T> using seq_concat=typename _impl::seq_concat_helper<T...>::type;

//----------------------------------------------------------------------------------------------------------------------

template<typename TInts, TInts ...N, typename TA>
std::ostream &seq_print(integer_sequence<TInts, N...>, std::ostream &os, TA const &d)
{
    size_type ndims = sizeof...(N);
    TInts dims[] = {N...};
    TInts idx[ndims];

    for (int i = 0; i < ndims; ++i)
    {
        idx[i] = 0;
    }

    while (1)
    {

        os << access(d, idx) << ", ";

        ++idx[ndims - 1];

        for (int rank = ndims - 1; rank > 0; --rank)
        {
            if (idx[rank] >= dims[rank])
            {
                idx[rank] = 0;
                ++(idx[rank - 1]);

                if (rank == ndims - 1)
                {
                    os << "\n";
                }
            }
        }
        if (idx[0] >= dims[0])
        {
            break;
        }
    }
    return os;
}

}// namespace traits

template<typename _Tp, _Tp First, _Tp ...Others>
std::ostream &operator<<(std::ostream &os, integer_sequence<_Tp, First, Others...> const &)
{
    os << First << " , " <<

       integer_sequence<_Tp, Others...>();

    return os;
}

template<typename _Tp, _Tp First>
std::ostream &operator<<(std::ostream &os, integer_sequence<_Tp, First> const &)
{
    os << First << std::endl;
    return os;
}

template<typename _Tp>
std::ostream &operator<<(std::ostream &os, integer_sequence<_Tp> const &)
{
    os << std::endl;
    return os;
}


template<size_type I> using index_const=std::integral_constant<size_type, I>;

}// namespace simpla
#endif /* CORE_toolbox_INTEGER_SEQUENCE_H_ */
