//
// Created by salmon on 16-12-22.
//

#ifndef SIMPLA_ARITHMETIC_H
#define SIMPLA_ARITHMETIC_H

#include <simpla/SIMPLA_config.h>
#include <cmath>
#include <type_traits>
#include "Algebra.h"
#include "Expression.h"

namespace simpla
{
namespace algebra
{


/** @name Constant Expressions
 * @{*/


template<typename value_type> struct Constant { value_type value; };
struct Zero {};
struct One {};
struct Infinity {};
struct Undefined {};
struct Identity{};

template<typename TE> inline TE const &operator+(TE const &e, Zero const &) { return (e); }

template<typename TE> inline TE const &operator+(Zero const &, TE const &e) { return (e); }

template<typename TE> inline TE const &operator-(TE const &e, Zero const &) { return (e); }

//template<typename TE> inline auto operator -(Zero const &, TE const &e)
//AUTO_RETURN (((-e)))

inline constexpr auto operator+(Zero const &, Zero const &e) AUTO_RETURN(Zero())

template<typename TE> inline constexpr auto operator*(TE const &e, One const &) AUTO_RETURN(e)

template<typename TE> inline constexpr auto operator*(One const &, TE const &e) AUTO_RETURN(e)

template<typename TE> inline constexpr auto operator*(TE const &, Zero const &) AUTO_RETURN(Zero())

template<typename TE> inline constexpr auto operator*(Zero const &, TE const &) AUTO_RETURN (Zero())

template<typename TE> inline constexpr auto operator/(TE const &e, Zero const &) AUTO_RETURN(Infinity())

template<typename TE> inline constexpr auto operator/(Zero const &, TE const &e) AUTO_RETURN(Zero())

template<typename TE> inline constexpr auto operator/(TE const &, Infinity const &) AUTO_RETURN(Zero())

template<typename TE> inline constexpr auto operator/(Infinity const &, TE const &e) AUTO_RETURN(Infinity())

//template<typename TL> inline auto operator==(TL const &lhs, Zero)AUTO_RETURN ((lhs))
//template<typename TR> inline auto operator==(Zero, TR const &rhs)AUTO_RETURN ((rhs))

constexpr auto operator&(Identity, Identity) AUTO_RETURN(Identity())

template<typename TL> constexpr auto operator&(TL const &l, Identity) AUTO_RETURN(l)

template<typename TR> constexpr auto operator&(Identity, TR const &r) AUTO_RETURN(r)

template<typename TL> constexpr auto operator&(TL const &l, Zero) AUTO_RETURN(std::move(Zero()))

template<typename TR> constexpr auto operator&(Zero, TR const &l) AUTO_RETURN(std::move(Zero()))

template<typename TR> constexpr auto operator&(Zero, Zero) AUTO_RETURN(std::move(Zero()))

/** @} */

#define DEF_BOP(_NAME_, _OP_)  \
 namespace tags{struct _NAME_{ \
     template<typename TL,typename TR> static inline constexpr auto eval( TL const & l,TR const & r) AUTO_RETURN( ( l _OP_ r) ) \
     template<typename TL,typename TR> inline constexpr auto operator()( TL const & l,TR const & r )const  AUTO_RETURN(  ( l  _OP_  r) ) \
};}

#define DEF_UOP(_NAME_, _OP_)   \
 namespace tags{struct _NAME_{  \
  template<typename TL> static inline constexpr auto eval(TL const & l ) AUTO_RETURN( (_OP_ l) ); \
  template<typename TL> inline constexpr auto operator()(TL const & l ) const  AUTO_RETURN( (_OP_ l) );  \
};}


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

#define DEF_BI_FUN(_NAME_)  \
namespace tags{struct _##_NAME_{ template<typename TL,typename TR> static inline constexpr auto  eval( TL const & l,TR const & r ) AUTO_RETURN(_NAME_(l , r));};}

#define DEF_UN_FUN(_NAME_)   \
namespace tags{struct _##_NAME_{ template<typename TL> static inline constexpr auto  eval(TL const & l ) AUTO_RETURN( _NAME_ (l));};}

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
//DEF_UN_FUN(real)
//DEF_UN_FUN(imag)

DEF_BI_FUN(atan2)
DEF_BI_FUN(pow)
#undef DEF_UN_FUN
#undef DEF_BI_FUN

namespace tags{struct _swap{template<typename TL, typename TR> static inline void eval(TL &l, TR &r){std::swap(l, r);};};}


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
namespace tags
{
struct _assign
{
    template<typename TL, typename TR> inline TL &operator()(TL &l, TR const &r) const
    {
        l = static_cast<TL>(r);
        return l;
    };

    template<typename TL, typename TR> static inline void eval(TL &l, TR const &r) { l = static_cast<TL>(r); };
};

struct _clear {};
struct _gather {};
struct _scatter {};
}

//#define DEF_ASSIGN_OP(_NAME_, _OP_)   \
//namespace tags{struct _NAME_##_assign{ template<typename TL,typename TR> static inline constexpr void eval( TL  & l,TR const & r){  l _OP_##= r;};};}
//
//DEF_ASSIGN_OP(,)
//DEF_ASSIGN_OP(plus, +)
//DEF_ASSIGN_OP(minus, -)
//DEF_ASSIGN_OP(multiplies, *)
//DEF_ASSIGN_OP(divides, /)
//DEF_ASSIGN_OP(modulus, %)
//
//#undef DEF_ASSIGN_OP


namespace declare
{
template<typename ...> struct Expression;
template<typename ...> struct BooleanExpression;
template<typename ...> struct AssignmentExpression;


#define _SP_DEFINE_EXPR_BINARY_OPERATOR(_OP_, _NAME_)                           \
    template< typename T1,typename T2> auto operator _OP_(T1   & l, T2    &r)           AUTO_RETURN( ( Expression< tags::_NAME_,  T1,  T2 > (l,r) )); \
    template< typename T1,typename T2> auto operator _OP_(T1  const & l, T2  const  &r) AUTO_RETURN((Expression< tags::_NAME_, const T1, const T2 > (l,r)))


#define _SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR(_OP_, _NAME_)                      \
    template< typename T1,typename T2> auto operator _OP_(T1   & l, T2    &r)           AUTO_RETURN( ( Expression< tags::_NAME_,  T1,  T2 > (l,r))) \
    template< typename T1,typename T2> auto operator _OP_(T1 const  & l, T2  const  &r) AUTO_RETURN( ( Expression< tags::_NAME_, const  T1, const T2 > (l,r)))


#define _SP_DEFINE_EXPR_UNARY_OPERATOR(_OP_, _NAME_)                           \
    template< typename T1> auto operator _OP_(T1   & l)      AUTO_RETURN((Expression< tags::_NAME_,  T1 > (l))) \
    template< typename T1> auto operator _OP_(T1  const & l) AUTO_RETURN((Expression< tags::_NAME_, const  T1 > (l)))


#define _SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(_OP_, _NAME_)                            \
    template< typename T1,typename T2> auto operator _OP_(T1   & l, T2    &r)           AUTO_RETURN((BooleanExpression< tags::_NAME_,   T1,  T2 > (l,r))) \
    template< typename T1,typename T2> auto operator _OP_(T1 const  & l, T2 const   &r) AUTO_RETURN((BooleanExpression< tags::_NAME_,  const T1, const T2 > (l,r)))


#define _SP_DEFINE_EXPR_UNARY_BOOLEAN_OPERATOR(_OP_, _NAME_)                           \
    template< typename T1> auto operator _OP_(T1   & l)     AUTO_RETURN((BooleanExpression< tags::_NAME_,  T1 > (l))) \
    template< typename T1> auto operator _OP_(T1  const & l)AUTO_RETURN((BooleanExpression< tags::_NAME_, const T1 > (l)))


#define _SP_DEFINE_EXPR_BINARY_FUNCTION(_NAME_)                                       \
    template< typename T1,typename T2> auto  _NAME_(T1   & l, T2    &r)        AUTO_RETURN ((Expression< tags::_##_NAME_,   T1,  T2 > (l,r)))    \
    template< typename T1,typename T2> auto  _NAME_(T1 const & l, T2 const &r) AUTO_RETURN ((Expression< tags::_##_NAME_,  const T1, const T2 > (l,r)))

#define _SP_DEFINE_EXPR_UNARY_FUNCTION(_NAME_)                                  \
    template< typename T1> auto  _NAME_(T1   & l)     AUTO_RETURN((Expression< tags::_##_NAME_,  T1 > (l))) \
    template< typename T1> auto  _NAME_(T1  const & l)AUTO_RETURN((Expression< tags::_##_NAME_, const T1 > (l)))


_SP_DEFINE_EXPR_BINARY_OPERATOR(+, plus)

_SP_DEFINE_EXPR_BINARY_OPERATOR(-, minus)

_SP_DEFINE_EXPR_BINARY_OPERATOR(*, multiplies)

_SP_DEFINE_EXPR_BINARY_OPERATOR(/, divides)

_SP_DEFINE_EXPR_BINARY_OPERATOR(%, modulus)

_SP_DEFINE_EXPR_BINARY_OPERATOR(^, bitwise_xor)

_SP_DEFINE_EXPR_BINARY_OPERATOR(&, bitwise_and)

_SP_DEFINE_EXPR_BINARY_OPERATOR(|, bitwise_or)

_SP_DEFINE_EXPR_UNARY_OPERATOR(~, bitwise_not)

_SP_DEFINE_EXPR_UNARY_OPERATOR(+, unary_plus)

_SP_DEFINE_EXPR_UNARY_OPERATOR(-, negate)

//_SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR(<<, shift_left)
//
//_SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR(>>, shift_right)

_SP_DEFINE_EXPR_UNARY_FUNCTION(cos)

_SP_DEFINE_EXPR_UNARY_FUNCTION(acos)

_SP_DEFINE_EXPR_UNARY_FUNCTION(cosh)

_SP_DEFINE_EXPR_UNARY_FUNCTION(sin)

_SP_DEFINE_EXPR_UNARY_FUNCTION(asin)

_SP_DEFINE_EXPR_UNARY_FUNCTION(sinh)

_SP_DEFINE_EXPR_UNARY_FUNCTION(tan)

_SP_DEFINE_EXPR_UNARY_FUNCTION(tanh)

_SP_DEFINE_EXPR_UNARY_FUNCTION(atan)

_SP_DEFINE_EXPR_UNARY_FUNCTION(exp)

_SP_DEFINE_EXPR_UNARY_FUNCTION(log)

_SP_DEFINE_EXPR_UNARY_FUNCTION(log10)

_SP_DEFINE_EXPR_UNARY_FUNCTION(sqrt)

_SP_DEFINE_EXPR_BINARY_FUNCTION(atan2)

_SP_DEFINE_EXPR_BINARY_FUNCTION(pow)

_SP_DEFINE_EXPR_UNARY_BOOLEAN_OPERATOR(!, logical_not)

_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(&&, logical_and)

_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(||, logical_or)

_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(!=, not_equal_to)

_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(==, equal_to)

_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(<, less)

_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(>, greater)

_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(<=, less_equal)

_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(>=, greater_equal)


//_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(real)                                          \
//_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(imag)                                          \


#undef _SP_DEFINE_EXPR_BINARY_OPERATOR
#undef _SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR
#undef _SP_DEFINE_EXPR_UNARY_OPERATOR
#undef _SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR
#undef _SP_DEFINE_EXPR_UNARY_BOOLEAN_OPERATOR
#undef _SP_DEFINE_EXPR_BINARY_FUNCTION
#undef _SP_DEFINE_EXPR_UNARY_FUNCTION


#define _SP_DEFINE_COMPOUND_OP(_OP_) template<typename TL, typename TR> inline TL & operator _OP_##=(TL &lhs, TR const &rhs){    lhs = lhs _OP_ rhs;    return lhs;}

_SP_DEFINE_COMPOUND_OP(+)

_SP_DEFINE_COMPOUND_OP(-)

_SP_DEFINE_COMPOUND_OP(*)

_SP_DEFINE_COMPOUND_OP(/)

#undef _SP_DEFINE_COMPOUND_OP
} // namespace declare
}
}//namespace simpla:: algebra
#endif //SIMPLA_ARITHMETIC_H
