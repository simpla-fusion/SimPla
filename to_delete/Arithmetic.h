//
// Created by salmon on 16-12-22.
//

#ifndef SIMPLA_ARITHMETIC_H
#define SIMPLA_ARITHMETIC_H

#include <simpla/SIMPLA_config.h>
#include <cmath>
#include <type_traits>
#include "simpla/algebra/Algebra.h"

namespace simpla {
namespace algebra {

/** @name Constant Expressions
 * @{*/

template <typename value_type>
struct Constant {
    value_type value;
};
struct Zero {};
struct One {};
struct Infinity {};
struct Undefined {};
struct Identity {};

template <typename TE>
TE const &operator+(TE const &e, Zero const &) {
    return (e);
}

template <typename TE>
TE const &operator+(Zero const &, TE const &e) {
    return (e);
}

template <typename TE>
TE const &operator-(TE const &e, Zero const &) {
    return (e);
}

// template<typename TE>  auto operator -(Zero const &, TE const &e)
// {return  (((-e)));}

constexpr auto operator+(Zero const &, Zero const &e) { return (Zero()); }

template <typename TE>
constexpr auto operator*(TE const &e, One const &) {
    return (e);
}
template <typename TE>
constexpr auto operator*(One const &, TE const &e) {
    return (e);
}
template <typename TE>
constexpr auto operator*(TE const &, Zero const &) {
    return (Zero());
}
template <typename TE>
constexpr auto operator*(Zero const &, TE const &) {
    return (Zero());
}
template <typename TE>
constexpr auto operator/(TE const &e, Zero const &) {
    return (Infinity());
}
template <typename TE>
constexpr auto operator/(Zero const &, TE const &e) {
    return (Zero());
}
template <typename TE>
constexpr auto operator/(TE const &, Infinity const &) {
    return (Zero());
}
template <typename TE>
constexpr auto operator/(Infinity const &, TE const &e) {
    return (Infinity());
}

// template<typename TL>  auto operator==(TL const &lhs, Zero){return  ((lhs));}
// template<typename TR>  auto operator==(Zero, TR const &rhs){return  ((rhs));}

constexpr auto operator&(Identity, Identity) { return (Identity()); }

template <typename TL>
constexpr auto operator&(TL const &l, Identity) {
    return (l);
}

template <typename TR>
constexpr auto operator&(Identity, TR const &r) {
    return (r);
}

template <typename TL>
constexpr auto operator&(TL const &l, Zero) {
    return (std::move(Zero()));
}

template <typename TR>
constexpr auto operator&(Zero, TR const &l) {
    return (std::move(Zero()));
}

template <typename TR>
constexpr auto operator&(Zero, Zero) {
    return (std::move(Zero()));
}

/** @} */

#define DEF_BOP(_NAME_, _OP_)                                       \
    namespace tags {                                                \
    struct _NAME_ {                                                 \
        template <typename TL, typename TR>                         \
        static constexpr auto eval(TL const &l, TR const &r) {      \
            return ((l _OP_ r));                                    \
        }                                                           \
        template <typename TL, typename TR>                         \
        constexpr auto operator()(TL const &l, TR const &r) const { \
            return ((l _OP_ r));                                    \
        }                                                           \
    };                                                              \
    }

#define DEF_UOP(_NAME_, _OP_)                          \
    namespace tags {                                   \
    struct _NAME_ {                                    \
        template <typename TL>                         \
        static constexpr auto eval(TL const &l) {      \
            return ((_OP_ l));                         \
        }                                              \
        template <typename TL>                         \
        constexpr auto operator()(TL const &l) const { \
            return ((_OP_ l));                         \
        }                                              \
    };                                                 \
    }

DEF_BOP(plus, +)
DEF_BOP(minus, -)
DEF_BOP(multiplies, *)
DEF_BOP(divides, /)
DEF_BOP(modulus, %)
DEF_UOP(negate, -)
DEF_UOP(unary_plus, +)
DEF_BOP(bitwise_and, &)
DEF_BOP(bitwise_or, |)
DEF_BOP(bitwise_xor, ^)
DEF_UOP(bitwise_not, ~)
DEF_BOP(shift_left, <<)
DEF_BOP(shift_right, >>)
DEF_UOP(logical_not, !)
DEF_BOP(logical_and, &&)
DEF_BOP(logical_or, ||)
DEF_BOP(not_equal_to, !=)
DEF_BOP(greater, >)
DEF_BOP(less, <)
DEF_BOP(greater_equal, >=)
DEF_BOP(less_equal, <=)
DEF_BOP(equal_to, ==)

#undef DEF_UOP
#undef DEF_BOP

using namespace std;

#define DEF_BI_FUN(_NAME_)                                     \
    namespace tags {                                           \
    struct _##_NAME_ {                                         \
        template <typename TL, typename TR>                    \
        static constexpr auto eval(TL const &l, TR const &r) { \
            return (_NAME_(l, r));                             \
        }                                                      \
    };                                                         \
    }

#define DEF_UN_FUN(_NAME_)                        \
    namespace tags {                              \
    struct _##_NAME_ {                            \
        template <typename TL>                    \
        static constexpr auto eval(TL const &l) { \
            return (_NAME_(l));                   \
        }                                         \
    };                                            \
    }

DEF_UN_FUN(cos)
DEF_UN_FUN(acos)
DEF_UN_FUN(cosh)
DEF_UN_FUN(sin)
DEF_UN_FUN(asin)
DEF_UN_FUN(sinh)
DEF_UN_FUN(tan)
DEF_UN_FUN(tanh)
DEF_UN_FUN(atan)
DEF_UN_FUN(exp)
DEF_UN_FUN(log)
DEF_UN_FUN(log10)
DEF_UN_FUN(sqrt)
// DEF_UN_FUN(real)
// DEF_UN_FUN(imag)

DEF_BI_FUN(atan2)
DEF_BI_FUN(pow)
#undef DEF_UN_FUN
#undef DEF_BI_FUN

namespace tags {
struct _swap {
    template <typename TL, typename TR>
    static void eval(TL &l, TR &r) {
        std::swap(l, r);
    };

    template <typename TL, typename TR>
    void operator()(TL &l, TR &r) const {
        std::swap(l, r);
    };
};
}

/**
 * ### Assignment Operator
 *
 *   Pseudo-Signature 	 				              | Semantics
 *  --------------------------------------------------|--------------
 *  `operator+=(GeoObject &,Expression const &)`      | Assign operation +
 *  `operator-=(GeoObject & ,Expression const &)`     | Assign operation -
 *  `operator/=(GeoObject & ,Expression const &)`     | Assign operation /
 *  `operator*=(GeoObject & ,Expression const &)`     | Assign operation *
 */
namespace tags {
struct _assign {
    template <typename TL, typename TR>
    TL &operator()(TL &l, TR const &r) const {
        l = (r);
        return l;
    };

    template <typename TL, typename TR>
    static void eval(TL &l, TR const &r) {
        l = static_cast<TL>(r);
    };
};

struct _clear {};
struct _gather {};
struct _scatter {};
}

//#define DEF_ASSIGN_OP(_NAME_, _OP_)   \
//namespace tags{struct _NAME_##_assign{ template<typename TL,typename TR> static  constexpr void eval( TL  & l,TR const & r){  l _OP_##= r;};};}
//
// DEF_ASSIGN_OP(,)
// DEF_ASSIGN_OP(plus, +)
// DEF_ASSIGN_OP(minus, -)
// DEF_ASSIGN_OP(multiplies, *)
// DEF_ASSIGN_OP(divides, /)
// DEF_ASSIGN_OP(modulus, %)
//
//#undef DEF_ASSIGN_OP

}
}  // namespace simpla:: algebra
#endif  // SIMPLA_ARITHMETIC_H
