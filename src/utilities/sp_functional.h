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
namespace _impl
{
struct binary_right
{
	template<typename TL, typename TR>
	TR const &operator()(TL const &, TR const & r) const
	{
		return r;
	}
};

#define DEF_BOP(_NAME_,_OP_)                                                               \
struct _NAME_                                                                             \
{                                                                                              \
	template<typename TL, typename TR>                                                         \
	constexpr auto operator()(TL const & l, TR const & r) const->decltype(l _OP_ r)                 \
	{  return l _OP_ r;   }                                                                                          \
};

#define DEF_UOP(_NAME_,_OP_)     \
struct _NAME_                                                                             \
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
DEF_UOP(unary_plus, +)

DEF_BOP(not_equal_to, !=)
DEF_BOP(greater, >)
DEF_BOP(less, <)
DEF_BOP(greater_equal, >=)
DEF_BOP(less_equal, <=)

DEF_BOP(logical_and, &&)
DEF_BOP(logical_or, ||)
DEF_UOP(logical_not, !)
DEF_BOP(bitwise_and, &)
DEF_BOP(bitwise_or, |)
DEF_BOP(bitwise_xor, ^)
DEF_UOP(bitwise_not, ~)

DEF_BOP(shift_left, <<)
DEF_BOP(shift_right, >>)

#undef DEF_UOP
#undef DEF_BOP

struct equal_to
{
	template<typename TL, typename TR>
	constexpr bool operator()(TL const & l, TR const & r) const
	{
		return l == r;
	}
	constexpr bool operator()(double l, double r) const
	{
		return std::abs(l - r) <= std::numeric_limits<double>::epsilon();
	}
};

// The few binary functions we miss.
struct _atan2
{
	template<typename _Tp>
	_Tp operator()(const _Tp& x, const _Tp& y) const
	{
		return atan2(x, y);
	}
};

struct _pow
{
	template<typename _Tp>
	_Tp operator()(const _Tp& x, const _Tp& y) const
	{
		return pow(x, y);
	}
};

// Implementations of unary functions applied to valarray<>s.

struct _abs
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::abs(__t);
	}
};

struct _cos
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::cos(__t);
	}
};

struct _acos
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::acos(__t);
	}
};

struct _cosh
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::cosh(__t);
	}
};

struct _sin
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::sin(__t);
	}
};

struct _asin
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::asin(__t);
	}
};

struct _sinh
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::sinh(__t);
	}
};

struct _tan
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::tan(__t);
	}
};

struct _atan
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::atan(__t);
	}
};

struct _tanh
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::tanh(__t);
	}
};

struct _exp
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::exp(__t);
	}
};

struct _log
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::log(__t);
	}
};

struct _log10
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::log10(__t);
	}
};

struct _sqrt
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::sqrt(__t);
	}
};

struct _real
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::real(__t);
	}
};
struct _imag
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::imag(__t);
	}
};
}  // namespace _impl
}  // namespace simpla

#endif /* SP_FUNCTIONAL_H_ */
