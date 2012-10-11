/*
 * arithmetic.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 * STUPID!!   DO NOT CHANGE THIS EXPRESSION TEMPLATES WITHOUT REALLY REALLY GOOD REASON!!!!!
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 */

#ifndef ARITHMETIC_H_
#define ARITHMETIC_H_
#include "fetl_defs.h"
#include "operation.h"
namespace simpla
{
using namespace fetl;

// Arithmetic
//-----------------------------------------

template<int IFORM, typename TVL, typename TLExpr> //
inline Field<IFORM, TVL, arithmetic::OpNegative<Field<IFORM, TVL, TLExpr> > >  //
operator -(Field<IFORM, TVL, TLExpr> const & lhs)
{
	return (Field<IFORM, TVL, arithmetic::OpNegative<Field<IFORM, TVL, TLExpr> > >(
			lhs));
}

//------------------------------------------------------------------------------------

template<int IFORM, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline Field<IFORM, typename arithmetic::OpAddition<TVL, TVR>::ValueType,
		arithmetic::OpAddition<Field<IFORM, TVL, TLExpr>,
				Field<IFORM, TVR, TRExpr> > >                                 //
operator +(Field<IFORM, TVL, TLExpr> const &lhs,
		Field<IFORM, TVR, TRExpr> const & rhs)
{

	return (Field<IFORM, typename arithmetic::OpAddition<TVL, TVR>::ValueType,
			arithmetic::OpAddition<Field<IFORM, TVL, TLExpr>,
					Field<IFORM, TVR, TRExpr> > >(lhs, rhs));
}
//------------------------------------------------------------------------------------

template<int IFORM, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline Field<IFORM, typename arithmetic::OpSubtraction<TVL, TVR>::ValueType,
		arithmetic::OpSubtraction<Field<IFORM, TVL, TLExpr>,
				Field<IFORM, TVR, TRExpr> > >                                 //
operator -(Field<IFORM, TVL, TLExpr> const &lhs,
		Field<IFORM, TVR, TRExpr> const & rhs)
{

	return (Field<IFORM,
			typename arithmetic::OpSubtraction<TVL, TVR>::ValueType,
			arithmetic::OpSubtraction<Field<IFORM, TVL, TLExpr>,
					Field<IFORM, TVR, TRExpr> > >(lhs, rhs));
}

//------------------------------------------------------------------------------------
#define DEFINE_MULTI(_TV_)                                                                                \
template<int IFORM, typename TVL, typename TLExpr>                                                        \
inline Field<IFORM, typename arithmetic::OpMultiplication<TVL, _TV_>::ValueType,                          \
		arithmetic::OpMultiplication<Field<IFORM, TVL, TLExpr>, _TV_> >                                   \
operator *(Field<IFORM, TVL, TLExpr> const &lhs, _TV_ const & rhs)                                        \
{                                                                                                         \
	return (Field<IFORM,                                                                                  \
			typename arithmetic::OpMultiplication<TVL, _TV_>::ValueType,                                  \
			arithmetic::OpMultiplication<Field<IFORM, TVL, TLExpr>, _TV_> >(lhs,                          \
			rhs));                                                                                        \
}                                                                                                         \
                                                                                                          \
template<int IFORM, typename TVR, typename TRExpr>                                                        \
inline Field<IFORM, typename arithmetic::OpMultiplication<_TV_, TVR>::ValueType,                          \
		arithmetic::OpMultiplication<_TV_, Field<IFORM, TVR, TRExpr> > >                                  \
operator *(_TV_ const & lhs, Field<IFORM, TVR, TRExpr> const & rhs)                                      \
{                                                                                                         \
	return (Field<IFORM,                                                                                  \
			typename arithmetic::OpMultiplication<_TV_, TVR>::ValueType,                                  \
			arithmetic::OpMultiplication<_TV_, Field<IFORM, TVR, TRExpr> > >(                             \
			lhs, rhs));                                                                                   \
}

DEFINE_MULTI(Real)
DEFINE_MULTI(Complex)
#undef  DEFINE_MULTI

template<int IFORM, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline Field<IFORM, typename arithmetic::OpMultiplication<TVL, TVR>::ValueType,
		arithmetic::OpMultiplication<Field<IFORM, TVL, TLExpr>,
				Field<IZeroForm, TVR, TRExpr> > >                             //
operator *(Field<IFORM, TVL, TLExpr> const &lhs,
		Field<IZeroForm, TVR, TRExpr> const & rhs)
{

	return (Field<IFORM,
			typename arithmetic::OpMultiplication<TVL, TVR>::ValueType,
			arithmetic::OpMultiplication<Field<IFORM, TVL, TLExpr>,
					Field<IZeroForm, TVR, TRExpr> > >(lhs, rhs));
}

template<int IFORM, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline Field<IFORM, typename arithmetic::OpMultiplication<TVL, TVR>::ValueType,
		arithmetic::OpMultiplication<Field<IZeroForm, TVL, TLExpr>,
				Field<IFORM, TVR, TRExpr> > >                                 //
operator *(Field<IZeroForm, TVL, TLExpr> const &lhs,
		Field<IFORM, TVR, TRExpr> const & rhs)
{

	return (Field<IFORM,
			typename arithmetic::OpMultiplication<TVL, TVR>::ValueType,
			arithmetic::OpMultiplication<Field<IZeroForm, TVL, TLExpr>,
					Field<IFORM, TVR, TRExpr> > >(lhs, rhs));
}

template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline Field<IZeroForm,
		typename arithmetic::OpMultiplication<TVL, TVR>::ValueType,
		arithmetic::OpMultiplication<Field<IZeroForm, TVL, TLExpr>,
				Field<IZeroForm, TVR, TRExpr> > >                             //
operator *(Field<IZeroForm, TVL, TLExpr> const &lhs,
		Field<IZeroForm, TVR, TRExpr> const & rhs)
{

	return (Field<IZeroForm,
			typename arithmetic::OpMultiplication<TVL, TVR>::ValueType,
			arithmetic::OpMultiplication<Field<IZeroForm, TVL, TLExpr>,
					Field<IZeroForm, TVR, TRExpr> > >(lhs, rhs));
}

template<int IFORM, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline Field<IFORM, typename arithmetic::OpDivision<TVL, TVR>::ValueType,
		arithmetic::OpDivision<Field<IFORM, TVL, TLExpr>,
				Field<IZeroForm, TVR, TRExpr> > >                             //
operator /(Field<IFORM, TVL, TLExpr> const &lhs,
		Field<IZeroForm, TVR, TRExpr> const & rhs)
{

	return (Field<IFORM, typename arithmetic::OpDivision<TVL, TVR>::ValueType,
			arithmetic::OpDivision<Field<IFORM, TVL, TLExpr>,
					Field<IZeroForm, TVR, TRExpr> > >(lhs, rhs));
}

template<int IFORM, typename TVL, typename TLExpr, typename TVR>
inline Field<IFORM, typename arithmetic::OpDivision<TVL, TVR>::ValueType,
		arithmetic::OpDivision<Field<IFORM, TVL, TLExpr>, TVR> >              //
operator /(Field<IFORM, TVL, TLExpr> const &lhs, TVR const & rhs)
{
	return (Field<IFORM, typename arithmetic::OpDivision<TVL, TVR>::ValueType,
			arithmetic::OpDivision<Field<IFORM, TVL, TLExpr>, TVR> >(lhs, rhs));
}
template<typename TVL, typename TVR, typename TRExpr>
inline Field<IZeroForm, typename arithmetic::OpDivision<TVL, TVR>::ValueType,
		arithmetic::OpDivision<TVL, Field<IZeroForm, TVR, TRExpr> > >         //
operator /(TVL const & lhs, Field<IZeroForm, TVR, TRExpr> const &rhs)
{
	return (Field<IZeroForm,
			typename arithmetic::OpDivision<TVL, TVR>::ValueType,
			arithmetic::OpDivision<TVL, Field<IZeroForm, TVR, TRExpr> > >(lhs,
			rhs));
}

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline Field<IZeroForm, typename arithmetic::OpDivision<TVL, TVR>::ValueType,
		arithmetic::OpDivision<nTuple<N, TVL, TLExpr>,
				Field<IZeroForm, TVR, TRExpr> > >             //
operator /(nTuple<N, TVL, TLExpr> const & lhs,
		Field<IZeroForm, TVR, TRExpr> const &rhs)
{
	return (Field<IZeroForm,
			typename arithmetic::OpDivision<TVL, TVR>::ValueType,
			arithmetic::OpDivision<nTuple<N, TVL, TLExpr>,
					Field<IZeroForm, TVR, TRExpr> > >(lhs, rhs));
}
//
//
//

//------------------------------------------------------------------------------------

template<int IFORM, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
typename arithmetic::OpMultiplication<TVL, TVR>::ValueType //
InnerProduct(Field<IFORM, TVL, TLExpr> const & lhs,
		Field<IFORM, TVR, TRExpr> const & rhs)
{
	typedef typename arithmetic::OpMultiplication<TVL, TVR>::ValueType ValueType;
	return (lhs.grid.InnerProduct(lhs, rhs));

}

//} // namespace fetl
}
// namespace simpla

#endif /* ARITHMETIC_H_ */
