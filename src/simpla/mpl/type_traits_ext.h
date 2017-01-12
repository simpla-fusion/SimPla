/**
 * @file type_traits_ext.h
 *
 * @date 2015-6-12
 * @author salmon
 */

#ifndef CORE_toolbox_TYPE_TRAITS_EXT_H_
#define CORE_toolbox_TYPE_TRAITS_EXT_H_

#include "simpla/concept/CheckConcept.h"
#include "macro.h"

namespace simpla {
/// \note  http://stackoverflow.com/questions/3913503/metaprogram-for-bit-counting
template <unsigned long N>
struct CountBits {
    static const unsigned long n = CountBits<N / 2>::n + 1;
};

template <>
struct CountBits<0> {
    static const unsigned long n = 0;
};

inline unsigned long count_bits(unsigned long s) {
    unsigned long n = 0;
    while (s != 0) {
        ++n;
        s = s >> 1;
    }
    return n;
}

template <typename T>
inline T *PointerTo(T &v) {
    return &v;
}

template <typename T>
inline T *PointerTo(T *v) {
    return v;
}

template <typename TV, typename TR>
inline TV TypeCast(TR const &obj) {
    return std::move(static_cast<TV>(obj));
}

template <int...>
class int_tuple_t;
// namespace _impl
//{
////******************************************************************************************************
//// Third-part code begin
//// ref: https://gitorious.org/redistd/redistd
//// Copyright Jonathan Wakely 2012
//// Distributed under the Boost Software License, Version 1.0.
//// (See accompanying file LICENSE_1_0.txt or copy at
//// http://www.boost.org/LICENSE_1_0.txt)
//
///// A type that represents a parameter pack of zero or more integers.
// template<unsigned ... Indices>
// struct index_tuple
//{
//	/// Generate an index_tuple with an additional element.
//	template<unsigned N>
//	using append = index_tuple<Indices..., N>;
//};
//
///// Unary metafunction that generates an index_tuple containing [0, Size)
// template<unsigned Size>
// struct make_index_tuple
//{
//	typedef typename make_index_tuple<Size - 1>::type::template append<Size - 1> type;
//};
//
//// Terminal case of the recursive metafunction.
// template<>
// struct make_index_tuple<0u>
//{
//	typedef index_tuple<> type;
//};
//
// template<typename ... Types>
// using to_index_tuple = typename make_index_tuple<sizeof...(Types)>::type;
//// Third-part code end
////******************************************************************************************************
//
//}// namespace _impl

HAS_MEMBER_FUNCTION(swap)

template <typename T>
typename std::enable_if<has_member_function_swap<T>::value, void>::type sp_swap(T &l, T &r) {
    l.swap(r);
}

template <typename T>
typename std::enable_if<!has_member_function_swap<T>::value, void>::type sp_swap(T &l, T &r) {
    std::swap(l, r);
}

template <typename TI>
auto ref(TI &it, ENABLE_IF(traits::is_iterator<TI>::value)) {
    return ((*it));
}

template <typename TI>
auto ref(TI &it, ENABLE_IF(!traits::is_iterator<TI>::value)) {
    return ((it));
}

template <typename>
struct result_of;

template <typename F, typename... Args>
struct result_of<F(Args...)> {
    typedef typename std::result_of<F(Args...)>::type type;
};

namespace _impl {

struct GetValue {
    template <typename TL, typename TI>
    constexpr auto operator()(TL const &v, TI const s) const {
        return ((traits::index(v, s)));
    }

    template <typename TL, typename TI>
    constexpr auto operator()(TL &v, TI const s) const {
        return ((traits::index(v, s)));
    }
};
}
// namespace _impl
template <typename...>
struct index_of;

template <typename TC, typename TI>
struct index_of<TC, TI> {
    typedef typename result_of<_impl::GetValue(TC, TI)>::type type;
};

HAS_MEMBER_FUNCTION(print)

template <typename TV>
auto sp_print(std::ostream &os, TV const &v) ->
    typename std::enable_if<has_member_function_print<TV const, std::ostream &>::value,
                            std::ostream &>::type {
    return v.print(os);
}

template <typename TV>
auto sp_print(std::ostream &os, TV const &v) ->
    typename std::enable_if<!has_member_function_print<TV const, std::ostream &>::value,
                            std::ostream &>::type {
    os << v;
    return os;
}

template <typename TI, TI L, TI R>
struct sp_max {
    static constexpr TI value = L > R ? L : R;
};

template <typename TI, TI L, TI R>
struct sp_min {
    static constexpr TI value = L < R ? L : R;
};
template <typename T>
struct sp_pod_traits {
    typedef T type;
};

template <typename _Signature>
class sp_result_of {
    typedef typename std::result_of<_Signature>::type _type;

   public:
    typedef typename sp_pod_traits<_type>::type type;
};

}  // namespace simpla

#endif /* CORE_toolbox_TYPE_TRAITS_EXT_H_ */
