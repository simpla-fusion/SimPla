//
// Created by salmon on 16-12-28.
//

#ifndef SIMPLA_CALCULATOR_H
#define SIMPLA_CALCULATOR_H

#include "Algebra.h"

namespace simpla { namespace algebra { namespace calculus
{
template<typename V>
struct calculator
{
    template<typename T> static constexpr inline T &
    get_value(T &v) { return v; };

    template<typename T, typename I0> static constexpr inline st::remove_all_extents_t<T, I0> &
    get_value(T &v, I0 const *s,
              ENABLE_IF((st::is_indexable<T, I0>::value))) { return get_value(v[*s], s + 1); };

    template<typename T, typename I0> static constexpr inline st::remove_all_extents_t<T, I0> &
    get_value(T &v, I0 const *s, ENABLE_IF((!st::is_indexable<T, I0>::value))) { return v; };
private:
    template<typename T, typename ...Args> static constexpr inline T &
    get_value_(std::integral_constant<bool, false> const &, T &v, Args &&...)
    {
        return v;
    }
//
//
//    template<typename T, typename I0, typename ...Idx> static constexpr inline st::remove_extents_t<T, I0, Idx...> &
//    get_value_(std::integral_constant<bool, true> const &, T &v, I0 const &s0, Idx &&...idx)
//    {
//        return get_value(v[s0], std::forward<Idx>(idx)...);
//    };
//public:
//    template<typename T, typename I0, typename ...Idx> static constexpr inline st::remove_extents_t<T, I0, Idx...> &
//    get_value(T &v, I0 const &s0, Idx &&...idx)
//    {
//        return get_value_(std::integral_constant<bool, st::is_indexable<T, I0>::value>(),
//                          v, s0, std::forward<Idx>(idx)...);
//    };
//
//    template<typename T, size_type N> static constexpr inline T &
//    get_value(declare::nTuple_<T, N> &v, size_type const &s0) { return v[s0]; };
//
//    template<typename T, size_type N> static constexpr inline T const &
//    get_value(declare::nTuple_<T, N> const &v, size_type const &s0) { return v[s0]; };
public:
    template<typename TOP, typename ...Others, size_type ... index, typename ...Idx> static constexpr inline auto
    _invoke_helper(declare::Expression<TOP, Others...> const &expr, index_sequence<index...>, Idx &&... s)
    DECL_RET_TYPE((TOP::eval(get_value(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)))

    template<typename TOP, typename   ...Others, typename ...Idx> static constexpr inline auto
    get_value(declare::Expression<TOP, Others...> const &expr, Idx &&... s)
    DECL_RET_TYPE((_invoke_helper(expr, index_sequence_for<Others...>(), std::forward<Idx>(s)...)))

    template<typename TOP, typename ...Others>
    static constexpr inline void apply(TOP const &op, declare::Array_<V, NDIMS> &lhs, Others &&...others)
    {
//        _detail::_apply(op, lhs, rhs);
    };

};


}}}//namespace simpla{namespace algebra{namespace calculus{
#endif //SIMPLA_CALCULATOR_H
