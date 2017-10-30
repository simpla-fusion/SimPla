//
// Created by salmon on 17-6-23.
//

#ifndef SIMPLA_UTILITY_H
#define SIMPLA_UTILITY_H

#include <utility>
namespace simpla {
namespace utility {
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
struct seq_max : public std::integer_sequence<Tp> {};

template <typename _Tp, _Tp... I>
struct seq_max<std::integer_sequence<_Tp, I...>>
    : public std::integral_constant<_Tp, _impl::_seq_max<_Tp, I...>::value> {};

template <typename Tp>
struct seq_min : public std::integer_sequence<Tp> {};

template <typename _Tp, _Tp... I>
struct seq_min<std::integer_sequence<_Tp, I...>>
    : public std::integral_constant<_Tp, _impl::_seq_min<_Tp, I...>::value> {};

template <bool... B>
struct logical_or;
template <bool B0>
struct logical_or<B0> : public std::integral_constant<bool, B0> {};
template <bool B0, bool... B>
struct logical_or<B0, B...> : public std::integral_constant<bool, B0 || logical_or<B...>::value> {};

template <bool... B>
struct logical_and;
template <bool B0>
struct logical_and<B0> : public std::integral_constant<bool, B0> {};
template <bool B0, bool... B>
struct logical_and<B0, B...> : public std::integral_constant<bool, B0 && logical_or<B...>::value> {};

template <typename T, T... B>
struct mt_max;
template <typename T, T B0>
struct mt_max<T, B0> : public std::integral_constant<T, B0> {};
template <typename T, T B0, T... B>
struct mt_max<T, B0, B...>
    : public std::integral_constant<T, ((B0 > (mt_max<T, B...>::value)) ? B0 : (mt_max<T, B...>::value))> {};

template <typename T, T... B>
struct mt_min;
template <typename T, T B0>
struct mt_min<T, B0> : public std::integral_constant<T, B0> {};
template <typename T, T B0, T... B>
struct mt_min<T, B0, B...>
    : public std::integral_constant<T, ((B0 < (mt_min<T, B...>::value)) ? B0 : (mt_min<T, B...>::value))> {};

//----------------------------------------------------------------------------------------------------------------------

template <typename>
struct seq_value;
template <typename _Tp, _Tp... N>
struct seq_value<std::integer_sequence<_Tp, N...>> {
    static constexpr _Tp value[] = {N...};
};
template <typename _Tp, _Tp... N>
constexpr _Tp seq_value<std::integer_sequence<_Tp, N...>>::value[];

//----------------------------------------------------------------------------------------------------------------------

template <int N, typename...>
struct seq_get;

template <int N, typename Tp, Tp M, Tp... I>
struct seq_get<N, std::integer_sequence<Tp, M, I...>> {
    static constexpr Tp value = seq_get<N - 1, std::integer_sequence<Tp, I...>>::value;
};

template <typename Tp, Tp M, Tp... I>
struct seq_get<0, std::integer_sequence<Tp, M, I...>> {
    static constexpr Tp value = M;
};

template <typename Tp>
struct seq_get<0, std::integer_sequence<Tp>> {
    static constexpr Tp value = 0;
};
template <int N, typename _Tp, _Tp... I>
_Tp get(std::integer_sequence<_Tp, I...>) {
    return seq_get<N, std::integer_sequence<_Tp, I...>>::value;
};
template <typename TP, TP... N>
struct seq_product;
template <typename TP, TP N0>
struct seq_product<TP, N0> : public std::integral_constant<TP, N0> {};
template <typename TP, TP N0, TP... N>
struct seq_product<TP, N0, N...> : public std::integral_constant<TP, N0 * seq_product<TP, N...>::value> {};

template <typename TP, TP... N>
struct seq_sum;
template <typename TP, TP N0>
struct seq_sum<TP, N0> : public std::integral_constant<TP, N0> {};
template <typename TP, TP N0, TP... N>
struct seq_sum<TP, N0, N...> : public std::integral_constant<TP, N0 + seq_sum<TP, N...>::value> {};
namespace _impl {
// TODO need implement max_integer_sequence, min_integer_sequence
template <size_t...>
struct _seq_for;

template <size_t M>
struct _seq_for<M> {
    template <typename TOP, typename... Args>
    static void eval(TOP const &op, Args &&... args) {
        op(access(std::forward<Args>(args), M - 1)...);
        _seq_for<M - 1>::eval(op, std::forward<Args>(args)...);
    }
};

template <>
struct _seq_for<0> {
    template <typename TOP, typename... Args>
    static void eval(TOP const &op, Args &&... args) {}
};

template <size_t M, size_t... N>
struct _seq_for<M, N...> {
    template <typename TOP, typename... Args>
    static inline void eval(TOP const &op, Args &&... args) {
        eval(op, std::integer_sequence<int>(), std::forward<Args>(args)...);
    }

    template <typename TOP, int... L, typename... Args>
    static inline void eval(TOP const &op, std::index_sequence<L...>, Args &&... args) {
        _seq_for<N...>::eval(op, std::index_sequence<L..., M>(), std::forward<Args>(args)...);

        _seq_for<M - 1, N...>::eval(op, std::index_sequence<L...>(), std::forward<Args>(args)...);
    }
};
}

// namespace _impl{
template <int N, typename... Args>
void seq_for(std::index_sequence<N>, Args &&... args) {
    _impl::_seq_for<N>::eval(std::forward<Args>(args)...);
}

template <size_t... N, typename... Args>
void seq_for(std::index_sequence<N...>, Args &&... args) {
    _impl::_seq_for<N...>::eval(std::forward<Args>(args)...);
}
//----------------------------------------------------------------------------------------------------------------------

namespace _impl {
template <size_t...>
struct _seq_reduce;

template <size_t M, size_t... N>
struct _seq_reduce<M, N...> {
    template <typename Reduction, size_t... L, typename... Args>
    static auto eval(Reduction const &reduction, std::index_sequence<L...>, Args &&... args) {
        return ((reduction(
            _seq_reduce<N...>::eval(reduction, std::index_sequence<L..., M>(), std::forward<Args>(args)...),
            _seq_reduce<M - 1, N...>::eval(reduction, std::index_sequence<L...>(), std::forward<Args>(args)...))));
    }

    template <typename Reduction, typename... Args>
    static auto eval(Reduction const &reduction, Args &&... args) {
        return (eval(reduction, std::integer_sequence<int>(), std::forward<Args>(args)...));
    }
};

template <size_t... N>
struct _seq_reduce<1, N...> {
    template <typename Reduction, size_t... L, typename... Args>
    static inline auto eval(Reduction const &reduction, std::index_sequence<L...>, Args &&... args) {
        return ((_seq_reduce<N...>::eval(reduction, std::index_sequence<L..., 1>(), std::forward<Args>(args)...)));
    }
};

template <>
struct _seq_reduce<> {
    template <typename Reduction, size_t... L, typename Args>
    static inline auto eval(Reduction const &, std::index_sequence<L...>, Args const &args) {
        return ((access(args, std::index_sequence<(L - 1)...>())));
    }
};
}

// namespace _impl
template <size_t... N, typename TOP, typename... Args>
auto seq_reduce(std::index_sequence<N...>, TOP const &op, Args &&... args) {
    return ((_impl::_seq_reduce<N...>::eval(op, std::forward<Args>(args)...)));
}
//----------------------------------------------------------------------------------------------------------------------

template <typename TI, TI... N, typename TOP>
void seq_for_each(std::integer_sequence<TI, N...>, TOP const &op) {
    int ndims = sizeof...(N);
    int idx[10];
    int dims[] = {N...};

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

//----------------------------------------------------------------------------------------------------------------------

namespace _impl {

/**
 *  cat two tuple/integer_sequence
 */
template <typename...>
struct seq_concat_helper;

template <typename _Tp, _Tp... _M>
struct seq_concat_helper<std::integer_sequence<_Tp, _M...>> {
    typedef std::integer_sequence<_Tp, _M...> type;
};

template <typename _Tp, _Tp... _M, _Tp... _N>
struct seq_concat_helper<std::integer_sequence<_Tp, _M...>, std::integer_sequence<_Tp, _N...>> {
    typedef std::integer_sequence<_Tp, _M..., _N...> type;
};

template <typename _Tp, _Tp... _M>
struct seq_concat_helper<std::integer_sequence<_Tp, _M...>, std::integer_sequence<_Tp>> {
    typedef std::integer_sequence<_Tp, _M...> type;
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
std::ostream &seq_print(std::integer_sequence<TInts, N...>, std::ostream &os, TA const &d) {
    int ndims = sizeof...(N);
    TInts dims[] = {N...};
    TInts idx[ndims];

    for (int i = 0; i < ndims; ++i) { idx[i] = 0; }

    //    while (1) {
    //        os << access(d, idx) << ", ";
    //
    //        ++idx[ndims - 1];
    //
    //        for (int rank = ndims - 1; rank > 0; --rank) {
    //            if (idx[rank] >= dims[rank]) {
    //                idx[rank] = 0;
    //                ++(idx[rank - 1]);
    //
    //                if (rank == ndims - 1) { os << "\n"; }
    //            }
    //        }
    //        if (idx[0] >= dims[0]) { break; }
    //    }
    return os;
}
inline constexpr int NSum() { return 0; };

template <typename T0>
auto NSum(T0 const &a0) {
    return a0;
};
template <typename T0, typename... Args>
auto NSum(T0 const &a0, Args &&... args) {
    return a0 + NSum(std::forward<Args>(args)...);
};

inline constexpr int NProduct() { return 1; };

template <typename T0>
auto NProduct(T0 const &a0) {
    return a0;
};
template <typename T0, typename... Args>
auto NProduct(T0 const &a0, Args &&... args) {
    return a0 * NProduct(std::forward<Args>(args)...);
};

template <typename T0>
auto NAnd(T0 const &a0) {
    return a0;
};
template <typename T0, typename... Args>
auto NAnd(T0 const &a0, Args &&... args) {
    return a0 && NAnd(std::forward<Args>(args)...);
};

template <typename T0>
auto NOr(T0 const &a0) {
    return a0;
};
template <typename T0, typename... Args>
auto NOr(T0 const &a0, Args &&... args) {
    return a0 || NOr(std::forward<Args>(args)...);
};

template <typename T0>
auto NMax(T0 const &a0) {
    return a0;
};
template <typename T0, typename... Args>
auto NMax(T0 const &a0, Args &&... args) {
    return std::max(a0, NMax(std::forward<Args>(args)...));
};
template <typename T0>
auto NMin(T0 const &a0) {
    return a0;
};
template <typename T0, typename... Args>
auto NMin(T0 const &a0, Args &&... args) {
    return std::min(a0, NMin(std::forward<Args>(args)...));
};
namespace detail {
template <typename T0, typename... Args>
auto NGetHelp(std::integral_constant<size_type, 0>, T0 const &a0, Args &&... args) {
    return a0;
};
template <size_type N, typename T0, typename... Args>
auto NGetHelp(std::integral_constant<size_type, N>, T0 const &a9, Args &&... args) {
    return NGetHelp(std::integral_constant<size_type, N - 1>(), std::forward<Args>(args)...);
};
}
template <size_type N, typename... Args>
auto NGet(Args &&... args) {
    static_assert(N < sizeof...(Args), "out of range!");
    return detail::NGetHelp(std::integral_constant<size_type, N>(), std::forward<Args>(args)...);
};

template <typename V>
auto sp_sqrt(V const &v) {
    return v * v;
}

}  // namespace utility

}  // namespace simpla
#endif  // SIMPLA_UTILITY_H
