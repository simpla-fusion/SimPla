/*
 * arithmetic.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 * STUPID!!   DO NOT CHANGE THIS EXPRESSION TEMPLATES WITHOUT doubleLY doubleLY GOOD REASON!!!!!
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 */

#ifndef ARITHMETIC_H_
#define ARITHMETIC_H_

#include <type_traits>
#include <utility>
#include <complex>

namespace simpla
{
#define DECL_RET_TYPE(_EXPR_) ->decltype((_EXPR_)){return (_EXPR_);}

template<typename T, typename IDX> inline
auto index(const T& f, IDX const & idx) DECL_RET_TYPE((f[idx]))

template<typename T, typename IDX> inline T index(const T f[], IDX const & idx)
{
	return (f[idx]);
}

template<typename IDX> inline double index(double v, IDX)
{
	return (v);
}

template<typename IDX> inline std::complex<double> index(std::complex<double> v,
		IDX)
{
	return (v);
}

namespace _impl
{

template<typename TOP, typename TL, typename TR> class BiOp
{
public:
	typename std::conditional<
			std::is_copy_constructible<TL>::value
					&& !(std::is_trivial<TL>::value
							&& sizeof(TL) > sizeof(int) * 3), TL, TL const &>::type l_;
	typename std::conditional<
			std::is_copy_constructible<TR>::value
					&& !(std::is_trivial<TR>::value
							&& sizeof(TR) > sizeof(int) * 3), TR, TR const &>::type r_;

	typedef BiOp<TOP, TL, TR> ThisType;

	typedef decltype(TOP::eval(index(l_,0),index(r_,0))) Value;

	BiOp(TL const & l, TR const &r) :
			l_(l), r_(r)
	{
	}

	BiOp(ThisType const &) =default;

	template<typename IDX> inline
	Value operator[](IDX const & idx)const
	{
		return (TOP::eval(index(l_,idx),index(r_,idx)));
	}

};
struct OpMultiplication
{
	template<typename TL, typename TR> inline static auto eval(TL const & l,
			TR const &r)
			DECL_RET_TYPE(l*r)
};
struct OpDivision
{
	template<typename TL, typename TR> inline static auto eval(TL const & l,
			TR const &r)
			DECL_RET_TYPE(l/r)
};

struct OpAddition
{
	template<typename TL, typename TR> inline static auto eval(TL const & l,
			TR const &r)
			DECL_RET_TYPE(l+r)
};
;
struct OpSubtraction
{
	template<typename TL, typename TR> inline static auto eval(TL const & l,
			TR const &r)
			DECL_RET_TYPE(l-r)
};
;
}  // namespace _impl

template<typename TL, typename TR> inline typename std::enable_if<
		!(std::is_arithmetic<TL>::value && std::is_arithmetic<TR>::value),
		_impl::BiOp<_impl::OpAddition, TL, TR> >::type //
operator +(TL const &lhs, TR const & rhs)
{
	return (_impl::BiOp<_impl::OpAddition, TL, TR>(lhs, rhs));
}
template<typename TL, typename TR> inline typename std::enable_if<
		!(std::is_arithmetic<TL>::value && std::is_arithmetic<TR>::value),
		_impl::BiOp<_impl::OpSubtraction, TL, TR> >::type //
operator -(TL const &lhs, TR const & rhs)
{
	return (_impl::BiOp<_impl::OpSubtraction, TL, TR>(lhs, rhs));
}
template<typename TL, typename TR> inline typename std::enable_if<
		!(std::is_arithmetic<TL>::value && std::is_arithmetic<TR>::value),
		_impl::BiOp<_impl::OpMultiplication, TL, TR> >::type //
operator *(TL const &lhs, TR const & rhs)
{
	return (_impl::BiOp<_impl::OpMultiplication, TL, TR>(lhs, rhs));
}
template<typename TL, typename TR> inline typename std::enable_if<
		!(std::is_arithmetic<TL>::value && std::is_arithmetic<TR>::value),
		_impl::BiOp<_impl::OpDivision, TL, TR> >::type //
operator /(TL const &lhs, TR const & rhs)
{
	return (_impl::BiOp<_impl::OpDivision, TL, TR>(lhs, rhs));
}

template<typename TL> inline auto //
operator -(TL const &lhs) DECL_RET_TYPE((-1.0*lhs))

}
// namespace simpla

#endif /* ARITHMETIC_H_ */
