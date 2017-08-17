/**
 * @file  pretty_stream.h
 *
 *  created on: 2013-11-29
 *      Author: salmon
 */

#ifndef PRETTY_STREAM_H_
#define PRETTY_STREAM_H_

#include <complex>
#include <cstddef>
#include <iomanip>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "simpla/utilities/type_traits.h"
#include "simpla/utilities/utility.h"
namespace simpla {
template <typename T, size_t... N>
std::ostream &printNd(std::ostream &os, T const &d, std::index_sequence<N...> const &,
                      ENABLE_IF((!traits::is_indexable<T, size_t>::value))) {
    os << d;
    return os;
}

template <typename T, size_t M, size_t... N>
std::ostream &printNd(std::ostream &os, T const &d, std::index_sequence<M, N...> const &,
                      ENABLE_IF((traits::is_indexable<T, size_t>::value))) {
    os << "{";
    printNd(os, d[0], std::index_sequence<N...>());
    for (size_t i = 1; i < M; ++i) {
        os << " , ";
        printNd(os, d[i], std::index_sequence<N...>());
    }
    os << "}";

    return os;
}

/**
 * @ingroup toolbox
 * @addtogroup fancy_print Fancy print
 * @{
 */

template <typename TV, typename TI>
inline TV const *printNdArray(std::ostream &os, TV const *v, int rank, TI const *d, bool is_first = true,
                              bool is_last = true, std::string const &left_brace = "{", std::string const &sep = ",",
                              std::string const &right_brace = "}", bool is_slow_first = true) {
    constexpr int ELE_NUM_PER_LINE = 10;
    if (rank == 1) {
        os << left_brace;
        for (int s = 0; s < d[0]; ++s) {
            if (s > 0) { os << sep; }
            if (s % ELE_NUM_PER_LINE == 0 && s != 0) { os << std::endl; }
            os << std::setw(10) << (*v);
            ++v;
        }
        os << right_brace;

    } else {
        os << left_brace;
        //        v = printNdArray(os, v, rank - 1, d + 1, left_brace, sep, right_brace);

        for (int s = 0; s < d[0]; ++s) {
            if (s > 0) {
                os << sep;
                if (rank > 1) { os << std::endl; }
            }
            v = printNdArray(os, v, rank - 1, d + 1, s == 0, s == d[0] - 1, left_brace, sep, right_brace);
        }
        os << right_brace;
        return (v);
    }
    //    if (is_last) { os << std::endl; }
    return v;
}

template <typename TX, typename TY, typename... Others>
std::istream &get_(std::istream &is, size_t num, std::map<TX, TY, Others...> &a) {
    for (size_t s = 0; s < num; ++s) {
        TX x;
        TY y;
        is >> x >> y;
        a.emplace(x, y);
    }
    return is;
}

template <typename TI>
std::ostream &ContainerOutPut1(std::ostream &os, TI const &ib, TI const &ie) {
    if (ib == ie) { return os; }

    TI it = ib;

    os << *it;

    size_t s = 0;
    ++it;
    for (; it != ie; ++it) {
        os << " , " << *it;

        ++s;
        if (s % 10 == 0) { os << std::endl; }
    }

    return os;
}

template <typename TI>
std::ostream &ContainerOutPut1(std::ostream &os, TI const *d, int num) {
    if (num == 0) { return os; }

    os << d[0];

    for (int s = 1; s < num; ++s) {
        os << " , " << d[s];

        if (s % 10 == 0) { os << std::endl; }
    }

    return os;
}

template <typename TI>
std::ostream &ContainerOutPut2(std::ostream &os, TI const &ib, TI const &ie) {
    if (ib == ie) { return os; }

    TI it = ib;

    os << it->first << "=" << it->second;

    ++it;

    for (; it != ie; ++it) { os << " , " << it->first << " = " << it->second << "\n"; }
    return os;
}

template <typename TI, typename TFUN>
std::ostream &ContainerOutPut3(std::ostream &os, TI const &ib, TI const &ie, TFUN const &fun) {
    if (ib == ie) { return os; }

    TI it = ib;

    fun(os, it);

    ++it;

    for (; it != ie; ++it) {
        os << " , ";
        fun(os, it);
    }
    return os;
}
}  // namespace simpla
namespace std {
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::complex<T> &tv) {
    os << "{" << tv.real() << "," << tv.imag() << "}";
    return (os);
}

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, std::pair<T1, T2> const &p) {
    os << p.first << " = { " << p.second << " }";
    return (os);
}

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, std::map<T1, T2> const &p) {
    for (auto const &v : p) os << v << "," << std::endl;
    return (os);
}

template <typename TV, typename... Others>
std::istream &operator>>(std::istream &is, std::vector<TV, Others...> &a) {
    for (auto it = a.begin(); it != a.end(); ++it) { is >> *it; }
    //    for (auto &v : a) { is >> v; }
    //	std::Duplicate(std::istream_iterator<TV>(is), std::istream_iterator<TV>(),
    // std::back_inserter(a));
    return is;
}

template <typename U, typename... Others>
std::ostream &operator<<(std::ostream &os, std::vector<U, Others...> const &d) {
    return simpla::ContainerOutPut1(os, &d[0], d.size());
}

template <typename U, typename... Others>
std::ostream &operator<<(std::ostream &os, std::list<U, Others...> const &d) {
    return simpla::ContainerOutPut1(os, d.begin(), d.end());
}

template <typename U, typename... Others>
std::ostream &operator<<(std::ostream &os, std::set<U, Others...> const &d) {
    return simpla::ContainerOutPut1(os, d.begin(), d.end());
}

template <typename U, typename... Others>
std::ostream &operator<<(std::ostream &os, std::multiset<U, Others...> const &d) {
    return simpla::ContainerOutPut1(os, d.begin(), d.end());
}

template <typename TX, typename TY, typename... Others>
std::ostream &operator<<(std::ostream &os, std::map<TX, TY, Others...> const &d) {
    return simpla::ContainerOutPut2(os, d.begin(), d.end());
}

template <typename TX, typename TY, typename... Others>
std::ostream &operator<<(std::ostream &os, std::multimap<TX, TY, Others...> const &d) {
    return simpla::ContainerOutPut2(os, d.begin(), d.end());
}
// template <typename T, int... M>
// std::ostream &operator<<(std::ostream &os, algebra::declare::nTuple_<T, M...> const &v) {
//    return algebra::_detail::printNd_(os, v.m_data_, int_sequence<M...>());
//}
namespace _impl {
template <typename... Args>
std::ostream &print_helper(std::ostream &os, std::tuple<Args...> const &v, std::integral_constant<int, 0>) {
    return os;
};

template <typename... Args, int N>
std::ostream &print_helper(std::ostream &os, std::tuple<Args...> const &v, std::integral_constant<int, N>) {
    os << " , " << std::get<sizeof...(Args) - N>(v);
    print_helper(os, v, std::integral_constant<int, N - 1>());
    return os;
};
}

template <typename T, typename... Args>
std::ostream &operator<<(std::ostream &os, std::tuple<T, Args...> const &v) {
    os << "{ " << std::get<0>(v);
    _impl::print_helper(os, v, std::integral_constant<int, sizeof...(Args)>());
    os << "}";

    return os;
};
}  // namespace std{
//
// template<typename T, typename ...Others>
// std::ostream &Print(std::ostream &os, T const &first, Others &&... others)
//{
//    os << first << " , ";
//
//    Print(os, std::forward<Others>(others)...);
//
//    return os;
//};
//
// template<typename T>
// std::ostream &Print(std::ostream &os, T const &v, traits::is_printable_t<T> *_p = nullptr)
//{
//    return v.Print(os, 1);
//}

/** @}*/

#endif /* PRETTY_STREAM_H_ */
