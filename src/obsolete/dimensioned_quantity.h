/*
 * dimensioned_quantity.h
 *
 *  Created on: 2012-8-3
 *      Author: salmon
 */

#ifndef DIMENSIONED_QUANTITY_H_
#define DIMENSIONED_QUANTITY_H_
#include <cmath>
#include <type_traits>
#include <utility>
#include "expression.h"
namespace simpla
{

template<int I0, int I1, int I2, int I3, int I4, int I5, int I6,
		typename TV = double>
class DimensionedQuantity
{
public:
	typedef DimensionedQuantity<I0, I1, I2, I3, I4, I5, I6, TV> ThisType;
	typedef TV Value;
	ReferenceTraits<Value> value;

	DimensionedQuantity(Value v) :
			value(v)
	{
	}
	~DimensionedQuantity()
	{

	}

	template<typename TR>
	bool & operator==(
			DimensionedQuantity<I0, I1, I2, I3, I4, I5, I6, TR> const &rhs)
	{
		return (value == rhs.value);
	}

	template<typename TR>
	ThisType & operator+=(
			DimensionedQuantity<I0, I1, I2, I3, I4, I5, I6, TR> const &rhs)
	{
		value += rhs.value;
	}

	template<typename TR>
	ThisType & operator-=(
			DimensionedQuantity<I0, I1, I2, I3, I4, I5, I6, TR> const &rhs)
	{
		value += rhs.value;
	}

	template<typename TR>
	ThisType & operator/=(TR const &rhs)
	{
		value /= rhs;
	}

	template<typename TR>
	ThisType & operator*=(TR const &rhs)
	{
		value *= rhs;
	}

};

template<typename T>
struct DimensionLessQuantityTraits
{
	typedef typename

	std::conditional<
			std::is_same<
					DimensionedQuantity<0, 0, 0, 0, 0, 0, 0, typename T::Value>,
					T>::value,

			typename T::Value, T>::type type;
};
template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TL,
		int R0, int R1, int R2, int R3, int R4, int R5, int R6, typename TR>
inline auto operator*(
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TL> const & lhs,
		DimensionedQuantity<R0, R1, R2, R3, R4, R5, R6, TR> const & rhs)

		->typename DimensionLessQuantityTraits<

		DimensionedQuantity<L0 + R0, L1 + R1, L2 + R2, L3 + R3, L4 + R4,
		L5 + R5, L6 + R6, decltype(lhs.value*rhs.value)>

		>::type
{
	return (typename DimensionLessQuantityTraits<
			DimensionedQuantity<L0 + R0, L1 + R1, L2 + R2, L3 + R3, L4 + R4,
					L5 + R5, L6 + R6, decltype(lhs.value*rhs.value)> >::type(
			lhs.value * rhs.value));
}

template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TL,
		int R0, int R1, int R2, int R3, int R4, int R5, int R6, typename TR>
inline auto operator/(
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TL> const & lhs,
		DimensionedQuantity<R0, R1, R2, R3, R4, R5, R6, TR> const & rhs)

		->typename

		DimensionLessQuantityTraits<

		DimensionedQuantity<L0 - R0, L1 - R1, L2 - R2, L3 - R3, L4 - R4,
		L5 - R5, L6 - R6, decltype(lhs.value/rhs.value) >

		>::type
{
	return (typename DimensionLessQuantityTraits<
			DimensionedQuantity<L0 - R0, L1 - R1, L2 - R2, L3 - R3, L4 - R4,
					L5 - R5, L6 - R6, decltype(lhs.value/rhs.value)> >::type(
			lhs.value / rhs.value));
}

template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TV>
inline auto operator+(
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TV> const & lhs,
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TV> const & rhs)
				DECL_RET_TYPE(
						(DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6,
								decltype(lhs.value + rhs.value)>( lhs.value + rhs.value ))

				)

template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TV>
inline auto operator-(
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TV> const & lhs,
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TV> const & rhs)
				DECL_RET_TYPE(
						(DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6,
								decltype(lhs.value - rhs.value)>( lhs.value - rhs.value ))

				)

template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TL,
		typename TR>
inline auto operator*(
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TL> const & lhs,
		TR const &rhs)
		DECL_RET_TYPE(
				(
						DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6,
						decltype(lhs.value*rhs)>(lhs.value * rhs)
				))

template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TL,
		typename TR>
inline auto operator*(TL const &lhs,
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TR> const & rhs)
				DECL_RET_TYPE(
						(
								DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, decltype(lhs*rhs.value)>(
										lhs * rhs.value)
						))
template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TL,
		typename TR>
inline auto operator/(
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TL> const & lhs,
		TR const &rhs)
				DECL_RET_TYPE(
						(
								DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, decltype(lhs.value/rhs)>(
										lhs.value / rhs)
						))

template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TL,
		typename TR>
inline auto operator/(TL const &lhs,
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TR> const & rhs)
				DECL_RET_TYPE(
						(
								DimensionedQuantity<-L0, -L1, -L2, -L3, -L4, -L5, -L6, decltype(lhs/rhs.value)>(
										lhs / rhs.value)
						))


} //namespace simpla
#endif /* DIMENSIONED_QUANTITY_H_ */
