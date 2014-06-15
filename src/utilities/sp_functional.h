/*
 * sp_functional.h
 *
 *  Created on: 2014年6月15日
 *      Author: salmon
 */

#ifndef SP_FUNCTIONAL_H_
#define SP_FUNCTIONAL_H_
#include <functional>
namespace std
{

#define DEF_BOP(_NAME_,_OP_)                                                               \
template<> struct _NAME_<void>                                                                             \
{                                                                                              \
	template<typename TL, typename TR>                                                         \
	constexpr auto operator()(TL const & l, TR const & r) const->decltype(l _OP_ r)                 \
	{  return l _OP_ r;   }                                                                                          \
};

#define DEF_UOP(_NAME_,_OP_)     \
template<>struct _NAME_<void>                                                                             \
{                                                                                              \
	template<typename TL >                                                         \
	constexpr auto operator()(TL const & l ) const->decltype(_OP_ l )                 \
	{  return  _OP_ l;   }                                                                                          \
};

DEF_BOP(plus, +)
DEF_BOP(minus, -)
DEF_BOP(multiplies, *)
DEF_BOP(divides, /)
DEF_BOP(modulus, %)
DEF_UOP(negate, -)

DEF_BOP(equal_to, ==)
DEF_BOP(not_equal_to, !=)
DEF_BOP(greater, >)
DEF_BOP(less, <)
DEF_BOP(greater_equal, >=)
DEF_BOP(less_equal, <=)

DEF_BOP(logical_and, &&)
DEF_BOP(logical_or, ||)
DEF_UOP(logical_not, !)
DEF_BOP(bit_and, &)
DEF_BOP(bit_or, |)
DEF_BOP(bit_xor, ^)

#undef DEF_UOP
#undef DEF_BOP
}  // namespace std

#endif /* SP_FUNCTIONAL_H_ */
