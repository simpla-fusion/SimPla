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
#include "typeconvert.h"
namespace simpla
{


// Arithmetic
//-----------------------------------------

template<typename TG, int IFORM, typename TLExpr> //
inline Field<TG, IFORM, _impl::OpNegative<Field<TG, IFORM, TLExpr> > >        //
operator -(Field<TG, IFORM, TLExpr> const & lhs)
{
	return (Field<TG, IFORM, _impl::OpNegative<Field<TG, IFORM, TLExpr> > >(lhs));
}

//------------------------------------------------------------------------------------

template<typename TG, int IFORM, typename TLExpr, typename TRExpr>
inline Field<TG, IFORM,
		_impl::OpAddition<Field<TG, IFORM, TLExpr>, Field<TG, IFORM, TRExpr> > >  //
operator +(Field<TG, IFORM, TLExpr> const &lhs,
		Field<TG, IFORM, TRExpr> const & rhs)
{

	return (Field<TG, IFORM,
			_impl::OpAddition<Field<TG, IFORM, TLExpr>, Field<TG, IFORM, TRExpr> > >(
			lhs, rhs));
}

//------------------------------------------------------------------------------------

template<typename TG, int IFORM, typename TLExpr, typename TRExpr>
inline Field<TG, IFORM,
		_impl::OpSubtraction<Field<TG, IFORM, TLExpr>, Field<TG, IFORM, TRExpr> > >  //
operator -(Field<TG, IFORM, TLExpr> const &lhs,
		Field<TG, IFORM, TRExpr> const & rhs)
{

	return (Field<TG, IFORM,
			_impl::OpSubtraction<Field<TG, IFORM, TLExpr>,
					Field<TG, IFORM, TRExpr> > >(lhs, rhs));
}

#define DEFINE_OP(TV)                                                                                  \
template<typename TG, typename TVExpr>                                                                 \
inline Field<TG, IZeroForm,                                                                            \
		_impl::OpAddition<Field<TG, IZeroForm, TVExpr>, TV> >                         \
operator +(Field<TG, IZeroForm, TVExpr> const &lhs, TV const & rhs)                                    \
{                                                                                                      \
                                                                                                       \
	return (Field<TG, IZeroForm,                                                                       \
			_impl::OpAddition<Field<TG, IZeroForm, TVExpr>, TV> >(                    \
			lhs, rhs));                                                                                \
}                                                                                                      \
                                                                                                       \
template<typename TG, typename TVExpr>                                                                 \
inline Field<TG, IZeroForm,                                                                            \
		_impl::OpAddition<TV, Field<TG, IZeroForm, TVExpr> > >                        \
operator +(TV const &lhs, Field<TG, IZeroForm, TVExpr> const & rhs)                                    \
{                                                                                                      \
                                                                                                       \
	return (Field<TG, IZeroForm,                                                                       \
			_impl::OpAddition<TV, Field<TG, IZeroForm, TVExpr> > >(                   \
			lhs, rhs));                                                                                \
}                                                                                                      \
template<typename TG, typename TVExpr>                                                                 \
inline Field<TG, IZeroForm,                                                                            \
		_impl::OpSubtraction<TV, Field<TG, IZeroForm, TVExpr> > >                     \
operator -(TV const &lhs, Field<TG, IZeroForm, TVExpr> const & rhs)                                    \
{                                                                                                      \
                                                                                                       \
	return (Field<TG, IZeroForm,                                                                       \
			_impl::OpSubtraction<TV,                                                  \
					Field<TG, IZeroForm, TVExpr> > >(lhs, rhs));                                       \
}                                                                                                      \
                                                                                                       \
template<typename TG, typename TVExpr>                                                                 \
inline Field<TG, IZeroForm,                                                                            \
		_impl::OpSubtraction<Field<TG, IZeroForm, TVExpr>, TV> >                      \
operator -(Field<TG, IZeroForm, TVExpr> const &lhs, TV const & rhs)                                    \
{                                                                                                      \
                                                                                                       \
	return (Field<TG, IZeroForm,                                                                       \
			_impl::OpSubtraction<Field<TG, IZeroForm, TVExpr>,                        \
					TV> >(lhs, rhs));                                                                  \
}                                                                                                      \

DEFINE_OP(Real)
DEFINE_OP(Complex)
#undef DEFINE_OP
//------------------------------------------------------------------------------------
template<typename TG, int IFORM, typename TLExpr>
inline Field<TG, IFORM, _impl::OpMultiplication<Field<TG, IFORM, TLExpr>, Real> >  //
operator *(Field<TG, IFORM, TLExpr> const &lhs, Real rhs)
{
	return (Field<TG, IFORM,
			_impl::OpMultiplication<Field<TG, IFORM, TLExpr>, Real> >(lhs, rhs));
}

template<typename TG, int IFORM, typename TRExpr>
inline Field<TG, IFORM, _impl::OpMultiplication<Real, Field<TG, IFORM, TRExpr> > >  //
operator *(Real lhs, Field<TG, IFORM, TRExpr> const & rhs)
{
	return (Field<TG, IFORM,
			_impl::OpMultiplication<Real, Field<TG, IFORM, TRExpr> > >(lhs, rhs));
}
#define DEFINE_MULTI(_TV_)                                                                                \
template<typename TG,int IFORM, typename TLExpr>                                                        \
inline Field<TG,IFORM,                           \
		_impl::OpMultiplication<Field<TG,IFORM,  TLExpr>, _TV_> >                                   \
operator *(Field<TG,IFORM,  TLExpr> const &lhs, _TV_ const & rhs)                                        \
{                                                                                                         \
	return (Field<TG,IFORM,                                  \
			_impl::OpMultiplication<Field<TG,IFORM,  TLExpr>, _TV_> >(lhs,                          \
			rhs));                                                                                        \
}                                                                                                         \
                                                                                                          \
template<typename TG,int IFORM,  typename TRExpr>                                                        \
inline Field<TG,IFORM,                           \
		_impl::OpMultiplication<_TV_, Field<TG,IFORM,  TRExpr> > >                                  \
operator *(_TV_ const & lhs, Field<TG,IFORM,  TRExpr> const & rhs)                                      \
{                                                                                                         \
	return (Field<TG,IFORM,                                  \
			_impl::OpMultiplication<_TV_, Field<TG,IFORM,  TRExpr> > >(                             \
			lhs, rhs));                                                                                   \
}

//DEFINE_MULTI(Real)
DEFINE_MULTI(Complex)
#undef  DEFINE_MULTI

template<typename TG, int IFORM, typename TLExpr, typename TRExpr>
inline Field<TG, IFORM,
		_impl::OpMultiplication<Field<TG, IFORM, TLExpr>,
				Field<TG, IZeroForm, TRExpr> > >                              //
operator *(Field<TG, IFORM, TLExpr> const &lhs,
		Field<TG, IZeroForm, TRExpr> const & rhs)
{

	return (Field<TG, IFORM,
			_impl::OpMultiplication<Field<TG, IFORM, TLExpr>,
					Field<TG, IZeroForm, TRExpr> > >(lhs, rhs));
}

template<typename TG, int IFORM, typename TLExpr, typename TRExpr>
inline Field<TG, IFORM,
		_impl::OpMultiplication<Field<TG, IZeroForm, TLExpr>,
				Field<TG, IFORM, TRExpr> > >                                  //
operator *(Field<TG, IZeroForm, TLExpr> const &lhs,
		Field<TG, IFORM, TRExpr> const & rhs)
{

	return (Field<TG, IFORM,
			_impl::OpMultiplication<Field<TG, IZeroForm, TLExpr>,
					Field<TG, IFORM, TRExpr> > >(lhs, rhs));
}

template<typename TG, typename TLExpr, typename TRExpr>
inline Field<TG, IZeroForm,
		_impl::OpMultiplication<Field<TG, IZeroForm, TLExpr>,
				Field<TG, IZeroForm, TRExpr> > >                              //
operator *(Field<TG, IZeroForm, TLExpr> const &lhs,
		Field<TG, IZeroForm, TRExpr> const & rhs)
{

	return (Field<TG, IZeroForm,
			_impl::OpMultiplication<Field<TG, IZeroForm, TLExpr>,
					Field<TG, IZeroForm, TRExpr> > >(lhs, rhs));
}

template<typename TG, int IFORM, typename TLExpr, typename TRExpr>
inline Field<TG, IFORM,
		_impl::OpDivision<Field<TG, IFORM, TLExpr>, Field<TG, IZeroForm, TRExpr> > >  //
operator /(Field<TG, IFORM, TLExpr> const &lhs,
		Field<TG, IZeroForm, TRExpr> const & rhs)
{

	return (Field<TG, IFORM,
			_impl::OpDivision<Field<TG, IFORM, TLExpr>,
					Field<TG, IZeroForm, TRExpr> > >(lhs, rhs));
}

template<typename TG, int IFORM, typename TLExpr, typename TVR>
inline Field<TG, IFORM, _impl::OpDivision<Field<TG, IFORM, TLExpr>, TVR> >    //
operator /(Field<TG, IFORM, TLExpr> const &lhs, TVR const & rhs)
{
	return (Field<TG, IFORM, _impl::OpDivision<Field<TG, IFORM, TLExpr>, TVR> >(
			lhs, rhs));
}
template<typename TG, typename TVL, typename TRExpr>
inline Field<TG, IZeroForm,
		_impl::OpDivision<TVL, Field<TG, IZeroForm, TRExpr> > >       //
operator /(TVL const & lhs, Field<TG, IZeroForm, TRExpr> const &rhs)
{
	return (Field<TG, IZeroForm,
			_impl::OpDivision<TVL, Field<TG, IZeroForm, TRExpr> > >(lhs, rhs));
}

template<int N, typename TLExpr, typename TG, typename TRExpr>
inline Field<TG, IZeroForm,
		_impl::OpDivision<nTuple<N, TLExpr>, Field<TG, IZeroForm, TRExpr> > >  //
operator /(nTuple<N, TLExpr> const & lhs,
		Field<TG, IZeroForm, TRExpr> const &rhs)
{
	return (Field<TG, IZeroForm,
			_impl::OpDivision<nTuple<N, TLExpr>, Field<TG, IZeroForm, TRExpr> > >(
			lhs, rhs));
}

} // namespace simpla

#endif /* ARITHMETIC_H_ */
