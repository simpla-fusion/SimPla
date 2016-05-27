/**
 * @file integer_sequence.h
 *
 *  Created on: 2014-9-26
 *      Author: salmon
 */

#ifndef CORE_GTL_INTEGER_SEQUENCE_H_
#define CORE_GTL_INTEGER_SEQUENCE_H_

#include <stddef.h>
#include "check_concept.h"

namespace simpla
{
//template<typename _Tp, _Tp ... _I> struct integer_sequence;

//////////////////////////////////////////////////////////////////////
/// integer_sequence
//////////////////////////////////////////////////////////////////////
#if __cplusplus >= 201402L
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
 */
 *
 *
template<typename _Tp, _Tp ... _I> struct integer_sequence;
template<int ... Ints> using index_sequence = integer_sequence<int, Ints...>;
template<typename _Tp, _Tp ... _I>
struct integer_sequence
{
private:
    static constexpr int m_size_ = sizeof...(_I);

public:
    typedef integer_sequence<_Tp, _I...> type;

    static constexpr int size()
    {
        return m_size_;
    }

};

template<typename _Tp>
struct integer_sequence<_Tp>
{

public:
    typedef integer_sequence<_Tp> type;

    static constexpr int size()
    {
        return 0;
    }

};

template<class T, T N>
using make_integer_sequence =typename _impl::gen_seq<T, N>::type;

template<int N>
using make_index_sequence = make_integer_sequence<int, N>;


#endif
namespace traits
{

template<typename T, typename TI>
auto &index(T &v, integer_sequence<TI>, FUNCTION_REQUIREMENT((is_indexable<T, TI>::value))) { return v; }

template<typename T, typename TI, TI M, TI ...N>
auto &index(T &v, integer_sequence<TI, M, N...>, FUNCTION_REQUIREMENT((is_indexable<T, TI>::value)))
{
    return index(v[M], integer_sequence<TI, N...>());
}

template<typename> struct seq_value;
template<typename _Tp, _Tp ...N>
struct seq_value<integer_sequence<_Tp, N...> > { static constexpr _Tp value[] = {N...}; };
template<typename _Tp, _Tp ...N>
constexpr _Tp seq_value<integer_sequence<_Tp, N...>>::value[];


template<size_t N, typename ...> struct seq_get;

template<size_t N, typename Tp, Tp M, Tp ...I>
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
//	typedef integer_sequence<T, mpl::max<T,N,M>::value ...> type;
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

//TODO need implement max_integer_sequence, min_integer_sequence
template<size_t...> struct _seq_for;

template<size_t M>
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
    template<typename TOP, typename ...Args> static inline void eval(TOP const &op, Args &&... args) { }
};

template<size_t M, size_t ...N>
struct _seq_for<M, N...>
{

    template<typename TOP, typename ...Args>
    static inline void eval(TOP const &op, Args &&... args)
    {
        eval(op, integer_sequence<size_t>(), std::forward<Args>(args)...);
    }

    template<typename TOP, size_t ...L, typename ...Args>
    static inline void eval(TOP const &op, integer_sequence<size_t, L...>,
                            Args &&... args)
    {
        _seq_for<N...>::eval(op, integer_sequence<size_t, L..., M>(),
                             std::forward<Args>(args)...);

        _seq_for<M - 1, N...>::eval(op, integer_sequence<size_t, L...>(),
                                    std::forward<Args>(args)...);
    }

};

template<size_t N, typename ...Args>
void seq_for(integer_sequence<size_t, N>, Args &&... args) { _seq_for<N>::eval(std::forward<Args>(args) ...); }

template<size_t ... N, typename ...Args>
void seq_for(integer_sequence<size_t, N...>, Args &&... args) { _seq_for<N...>::eval(std::forward<Args>(args) ...); }

template<size_t...> struct _seq_reduce;

template<size_t M, size_t ...N>
struct _seq_reduce<M, N...>
{

    template<typename Reduction, size_t ...L, typename ... Args>
    static inline auto eval(Reduction const &reduction,
                            integer_sequence<size_t, L...>, Args &&... args)
    {
        return reduction(
                _seq_reduce<N...>::eval(reduction,
                                        integer_sequence<size_t, L..., M>(),
                                        std::forward<Args>(args)...),

                _seq_reduce<M - 1, N...>::eval(reduction,
                                               integer_sequence<size_t, L...>(),
                                               std::forward<Args>(args)...)

        );
    }

    template<typename Reduction, typename ...Args>
    static inline auto eval(Reduction const &reduction, Args &&... args)
    {
        return eval(reduction, integer_sequence<size_t>(), std::forward<Args>(args)...);
    }

};

template<size_t ...N>
struct _seq_reduce<1, N...>
{

    template<typename Reduction, size_t ...L, typename ...Args>
    static inline auto eval(Reduction const &reduction, integer_sequence<size_t, L...>,
                            Args &&... args)
    {
        return _seq_reduce<N...>::eval(reduction,
                                       integer_sequence<size_t, L..., 1>(),
                                       std::forward<Args>(args)...);
    }

};

template<>
struct _seq_reduce<>
{
    template<typename Reduction, size_t ...L, typename Args>
    static inline auto eval(Reduction const &, integer_sequence<size_t, L...>,
                            Args const &args)
    {
        return access((args), integer_sequence<size_t, (L - 1)...>());
    }

};

template<size_t ... N, typename TOP, typename ...Args>
auto seq_reduce(integer_sequence<size_t, N...>, TOP const &op, Args &&... args)
{
    return _seq_reduce<N...>::eval(op, std::forward<Args>(args)...);
}

template<typename TInts, TInts ...N, typename TOP>
void seq_for_each(integer_sequence<TInts, N...>, TOP const &op)
{
    size_t ndims = sizeof...(N);
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

template<typename TInts, TInts ...N, typename TA>
std::ostream &seq_print(integer_sequence<TInts, N...>, std::ostream &os, TA const &d)
{
    size_t ndims = sizeof...(N);
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

template<typename ...> struct seq_max;

template<typename U, typename ...T>
struct seq_max<U, T...> { typedef typename seq_max<U, seq_max<T...> >::type type; };
template<> struct seq_max<> { typedef void type; };
template<typename U> struct seq_max<U> { typedef U type; };
template<typename U> struct seq_max<U, void> { typedef U type; };
template<typename _Tp, _Tp ...N, _Tp ...M>
struct seq_max<integer_sequence<_Tp, N...>, integer_sequence<_Tp, M...>> { typedef integer_sequence<_Tp, N...> type; };
template<typename U, typename T>
struct seq_max<U, T> { typedef typename seq_max<U, seq_max<T> >::type type; };
template<typename ...> struct seq_min;
template<> struct seq_min<> { typedef void type; };
template<typename U> struct seq_min<U> { typedef U type; };
template<typename U> struct seq_min<U, void> { typedef U type; };
template<typename U, typename ...T> struct seq_min<U, T...> { typedef typename seq_min<U, seq_min<T...> >::type type; };


/**
 *  cat two tuple/integer_sequence
 */
template<typename ...> struct seq_concat;

template<typename _Tp, _Tp ... _M>
struct seq_concat<integer_sequence<_Tp, _M...>> : public integer_sequence<_Tp, _M...> { };

//template<typename _Tp, _Tp ..._M, typename ...Others>
//struct seq_concat<integer_sequence<_Tp, _M...>, Others...>
//        : public seq_concat<integer_sequence<_Tp, _M...>, seq_concat<Others...>>
//{
//};

template<typename _Tp, _Tp ... _M, _Tp ... _N>
struct seq_concat<integer_sequence<_Tp, _M...>, integer_sequence<_Tp, _N...> >
        : public integer_sequence<_Tp, _M..., _N...>
{
};
template<typename _Tp, _Tp ... _M>
struct seq_concat<integer_sequence<_Tp, _M...>, integer_sequence<_Tp> > : public integer_sequence<_Tp, _M...>
{
};

}// namespace _impl

template<typename _Tp, _Tp First, _Tp...Others>
std::ostream &operator<<(std::ostream &os, integer_sequence<_Tp, First, Others...> const &)
{
    os << First << " , " << integer_sequence<_Tp, Others...>();
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
}// namespace simpla
#endif /* CORE_GTL_INTEGER_SEQUENCE_H_ */
