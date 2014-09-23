/*
 * sp_functional.h
 *
 *  created on: 2014-6-15
 *      Author: salmon
 */

#ifndef SP_FUNCTIONAL_H_
#define SP_FUNCTIONAL_H_
#include <functional>
namespace simpla
{

#define DEF_BOP(_NAME_,_OP_)                                                               \
template<typename T=void> struct _NAME_                                                                             \
{                                                                                              \
	template<typename TL, typename TR>                                                         \
	constexpr auto operator()(TL const & l, TR const & r) const->decltype(l _OP_ r)                 \
	{  return l _OP_ r;   }                                                                                          \
};

#define DEF_UOP(_NAME_,_OP_)     \
template<typename T=void>struct _NAME_                                                                             \
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

template<typename T = void> struct op_imag
{
	template<typename T>
	constexpr T operator()(std::complex<T> const & l) const
	{
		return std::imag(l);
	}
};
template<typename T = void> struct op_real
{
	template<typename T>
	constexpr T operator()(std::complex<T> const & l) const
	{
		return std::real(l);
	}
};
template<> struct equal_to<double>
{
	constexpr bool operator()(double l, double r) const
	{
		return std::abs(l - r) <= std::numeric_limits<double>::epsilon();
	}
};
}  // namespace simpla

#endif /* SP_FUNCTIONAL_H_ */
