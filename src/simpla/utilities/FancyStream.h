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

/**
 * @ingroup toolbox
 * @addtogroup fancy_print Fancy print
 * @{
 */

template <typename TV, typename TI>
inline TV const *printNdArray(std::ostream &os, TV const *v, int rank, TI const *d, bool is_first = true,
                              bool is_last = true, std::string const &left_brace = "{", std::string const &sep = ",",
                              std::string const &right_brace = "}", bool is_slow_first = true, int tab_width = 0,
                              int indent = 0) {
    constexpr int ELE_NUM_PER_LINE = 10;
    if (rank == 1) {
        os << left_brace;
        for (int s = 0; s < d[0]; ++s) {
            if (s > 0) { os << sep; }
            if (s % ELE_NUM_PER_LINE == 0 && s != 0) { os << std::endl; }
            if (tab_width > 0) { os << std::setw(10); }
            os << (*v);
            ++v;
        }
        os << right_brace;

    } else {
        os << left_brace;

        for (int s = 0; s < d[0]; ++s) {
            if (s > 0) {
                os << sep;
                if (rank > 1) { os << std::endl << std::setw(indent); }
            }
            v = printNdArray(os, v, rank - 1, d + 1, s == 0, s == d[0] - 1, left_brace, sep, right_brace, is_slow_first,
                             tab_width, indent + 1);
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
template <typename V>
std::ostream &FancyPrint(std::ostream &os, V const *d, size_type ndims, size_type const *extents, int indent) {
    printNdArray(os, d, ndims, extents, true, false, "[", ",", "]", true, 0, indent);
    return os;
}

template <typename TI>
std::ostream &PrintArray1(std::ostream &os, TI const &ib, TI const &ie, int indent = 0);

template <typename TI>
std::ostream &PrintArray1(std::ostream &os, TI const *d, int num, int indent = 0);

template <typename TI>
std::ostream &PrintKeyValue(std::ostream &os, TI const &ib, TI const &ie, int indent = 0);

template <typename TI, typename TFUN>
std::ostream &ContainerOutPut3(std::ostream &os, TI const &ib, TI const &ie, TFUN const &fun, int indent = 0);

template <typename T>
std::ostream &FancyPrint(std::ostream &os, T const &d, int indent, ENABLE_IF(std::rank<T>::value == 0)) {
    os << d;
    return os;
}
template <typename T>
std::ostream &FancyPrint(std::ostream &os, T const &d, int indent, ENABLE_IF((std::rank<T>::value > 0))) {
    os << "[";
    FancyPrint(os, d[0], indent + 1);
    for (size_t i = 1; i < std::extent<T, 0>::value; ++i) {
        os << ", ";
        if (std::rank<T>::value > 1) { os << std::endl << std::setw(indent) << " "; }
        FancyPrint(os, d[i], indent + 1);
    }
    os << "]";

    return os;
}

inline std::ostream &FancyPrint(std::ostream &os, std::string const &s, int indent) {
    os << "\"" << s << "\"";
    return os;
}

template <typename T>
std::ostream &FancyPrint(std::ostream &os, const std::complex<T> &tv, int indent = 0) {
    os << "{" << tv.real() << "," << tv.imag() << "}";
    return (os);
}

template <typename T1, typename T2>
std::ostream &FancyPrint(std::ostream &os, std::pair<T1, T2> const &p, int indent = 0) {
    os << p.first << " = { " << p.second << " }";
    return (os);
}

template <typename T1, typename T2>
std::ostream &FancyPrint(std::ostream &os, std::map<T1, T2> const &p, int indent = 0) {
    for (auto const &v : p) os << v << "," << std::endl;
    return (os);
}

template <typename U, typename... Others>
std::ostream &FancyPrint(std::ostream &os, std::vector<U, Others...> const &d, int indent = 0) {
    os << "[ ";
    PrintArray1(os, d.begin(), d.end(), indent);
    os << " ]";
    return os;
}

template <typename U, typename... Others>
std::ostream &FancyPrint(std::ostream &os, std::list<U, Others...> const &d, int indent = 0) {
    return PrintArray1(os, d.begin(), d.end(), indent);
}

template <typename U, typename... Others>
std::ostream &FancyPrint(std::ostream &os, std::set<U, Others...> const &d, int indent = 0) {
    return PrintArray1(os, d.begin(), d.end(), indent);
}

template <typename U, typename... Others>
std::ostream &FancyPrint(std::ostream &os, std::multiset<U, Others...> const &d, int indent = 0) {
    return PrintArray1(os, d.begin(), d.end(), indent);
}

template <typename TX, typename TY, typename... Others>
std::ostream &FancyPrint(std::ostream &os, std::map<TX, TY, Others...> const &d, int indent = 0) {
    return PrintKeyValue(os, d.begin(), d.end(), indent);
}

template <typename TX, typename TY, typename... Others>
std::ostream &FancyPrint(std::ostream &os, std::multimap<TX, TY, Others...> const &d, int indent = 0) {
    return PrintKeyValue(os, d.begin(), d.end(), indent);
}
// template <typename T, int... M>
// std::ostream &FancyPrint(std::ostream &os, algebra::declare::nTuple_<T, M...> const &v) {
//    return algebra::_detail::printNd_(os, v.m_data_, int_sequence<M...>());
//}
namespace _impl {
template <typename... Args>
std::ostream &print_helper(std::ostream &os, std::tuple<Args...> const &v, std::integral_constant<int, 0>,
                           int indent = 0) {
    return os;
};

template <typename... Args, int N>
std::ostream &print_helper(std::ostream &os, std::tuple<Args...> const &v, std::integral_constant<int, N>,
                           int indent = 0) {
    os << ", ";
    FancyPrint(os, std::get<sizeof...(Args) - N>(v), indent);
    print_helper(os, v, std::integral_constant<int, N - 1>(), indent);
    return os;
};
}

template <typename T, typename... Args>
std::ostream &FancyPrint(std::ostream &os, std::tuple<T, Args...> const &v, int indent = 0) {
    os << "{ ";
    FancyPrint(os, std::get<0>(v), indent);
    _impl::print_helper(os, v, std::integral_constant<int, sizeof...(Args)>(), indent + 1);
    os << "}";

    return os;
};

template <typename TI>
std::ostream &PrintArray1(std::ostream &os, TI const &ib, TI const &ie, int indent) {
    if (ib == ie) { return os; }

    TI it = ib;
    FancyPrint(os, *it, indent + 1);
    size_t s = 0;
    ++it;
    for (; it != ie; ++it) {
        os << ", ";
        FancyPrint(os, *it, indent + 1);
        ++s;
        if (s % 10 == 0) { os << std::endl; }
    }

    return os;
}

// template <typename TI>
// std::ostream &PrintArry1(std::ostream &os, TI const *d, int num, int indent) {
//    if (num == 0) { return os; }
//    FancyPrint(os, d[0], indent + 1);
//    for (int s = 1; s < num; ++s) {
//        os << ", ";
//        FancyPrint(os, d[s], indent + 1);
//        if (s % 10 == 0) { os << std::endl; }
//    }
//
//    return os;
//}

template <typename TI>
std::ostream &PrintKeyValue(std::ostream &os, TI const &ib, TI const &ie, int indent) {
    if (ib == ie) { return os; }
    TI it = ib;
    FancyPrint(os, it->first, indent);
    os << "=";
    FancyPrint(os, it->second, indent + 1);
    ++it;
    for (; it != ie; ++it) {
        os << " , " << std::endl << std::setw(indent) << " ";
        FancyPrint(os, it->first, indent);
        os << " = ";
        FancyPrint(os, it->second, indent + 1);
    }
    return os;
}

// template <typename TI, typename TFUN>
// std::ostream &ContainerOutPut3(std::ostream &os, TI const &ib, TI const &ie, TFUN const &fun, int indent) {
//    if (ib == ie) { return os; }
//    TI it = ib;
//    fun(os, it);
//    ++it;
//    for (; it != ie; ++it) {
//        os << " , ";
//        fun(os, it);
//    }
//    return os;
//}
}  // namespace simpla
namespace std {
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::complex<T> &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, std::pair<T1, T2> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, std::map<T1, T2> const &v) {
    return simpla::FancyPrint(os, v, 0);
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
std::ostream &operator<<(std::ostream &os, std::vector<U, Others...> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename U, typename... Others>
std::ostream &operator<<(std::ostream &os, std::list<U, Others...> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename U, typename... Others>
std::ostream &operator<<(std::ostream &os, std::set<U, Others...> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename U, typename... Others>
std::ostream &operator<<(std::ostream &os, std::multiset<U, Others...> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename TX, typename TY, typename... Others>
std::ostream &operator<<(std::ostream &os, std::map<TX, TY, Others...> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename TX, typename TY, typename... Others>
std::ostream &operator<<(std::ostream &os, std::multimap<TX, TY, Others...> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename T, typename... Args>
std::ostream &operator<<(std::ostream &os, std::tuple<T, Args...> const &v) {
    return simpla::FancyPrint(os, v, 0);
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
