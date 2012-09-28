/*
 * dimensioned_field.h
 *
 *  Created on: 2012-3-18
 *      Author: salmon
 */

#ifndef DIMENSIONED_FIELD_H_
#define DIMENSIONED_FIELD_H_
#include "physics/physics.h"
#include "fetl/fetl_defs.h"
#include "fetl/fields.h"
namespace simpla
{

template<int IS, int I0, int I1, int I2, int I3, int I4, int I5, int I6,
		int IFORM, typename TV, typename TG>
struct Field<IFORM,
		physics::PhysicalQuantity<IS, I0, I1, I2, I3, I4, I5, I6, TV>
		, _FETL_Field<TG> > : public Field<IFORM, TV, _FETL_Field<TG> >
{
	typedef physics::PhysicalQuantity<IS, I0, I1, I2, I3, I4, I5, I6, TV> PhysicalValue;

	typedef Field<IFORM, PhysicalValue, _FETL_Field<TG> > ThisType;

	typedef Field<IFORM, TV, _FETL_Field<TG> > BaseType;

	Field(TG const & pgrid) :
			BaseType(pgrid), dimension_(1.0)
	{

	}
	~Field()
	{
	}

	inline PhysicalValue const & get_dimension() const
	{
		return (dimension_);
	}

	ThisType const & operator=(PhysicalValue const &rhs)
	{
		BaseType::operator=(rhs.value());
		return (*this);
	}
	template<typename TRV, typename TR>
	ThisType const & operator=(
			Field<
					IFORM,
					physics::PhysicalQuantity<IS, I0, I1, I2, I3, I4, I5, I6,
							TRV>, TR> const &rhs)
	{
		PhysicalValue dimsions(1.0);
		BaseType::operator=(rhs / dimsions);
		return (*this);
	}

	inline PhysicalValue operator[](size_t s) const
	{
		return (BaseType::operator[](s) * dimension_);

	}
	const PhysicalValue dimension_;
};

//
//template<int IS, int I0, int I1, int I2, int I3, int I4, int I5, int I6,
//		int IFORM, typename TV, typename TL, typename TR>
//inline typename BiOp<Field<IFORM, TV, TL>
//		,physics::PhysicalQuantity<IS, I0, I1, I2, I3, I4, I5, I6, TR>
//		,arithmetic::OpMultiplication>::ResultType //
//operator *(
//		Field<IFORM, TV, TL> const &lhs
//		, physics::PhysicalQuantity<IS, I0, I1, I2, I3, I4, I5, I6, TR> const & rhs)
//{
//	typedef BiOp<Field<IFORM, TV, TL>
//			,physics::PhysicalQuantity<IS, I0, I1, I2, I3, I4, I5, I6, TR>
//			,arithmetic::OpMultiplication> TOP;
//	return (typename TOP::ResultType(TOP(lhs, rhs)));
//}
//
//template<int IS, int I0, int I1, int I2, int I3, int I4, int I5, int I6,
//		typename TL, int IFORM, typename TV, typename TR>
//inline typename BiOp<
//		physics::PhysicalQuantity<IS, I0, I1, I2, I3, I4, I5, I6, TR>
//		, Field<IFORM, TV, TR> ,arithmetic::OpMultiplication>::ResultType //
//operator *(
//		physics::PhysicalQuantity<IS, I0, I1, I2, I3, I4, I5, I6, TR> const & lhs
//		, Field<IFORM, TV, TR> const &rhs)
//{
//	typedef BiOp<physics::PhysicalQuantity<IS, I0, I1, I2, I3, I4, I5, I6, TR>
//			, Field<IFORM, TV, TR> ,arithmetic::OpMultiplication> TOP;
//	return (typename TOP::ResultType(TOP(lhs, rhs)));
//}

}// namespace simpla

#endif /* DIMENSIONED_FIELD_H_ */
