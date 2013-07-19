/*
 * fetl_defs.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 * STUPID!!   DO NOT CHANGE THIS EXPRESSION TEMPLATES WITHOUT REALLY REALLY GOOD REASON!!!!!
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

#ifndef FETL_DEFS_H_
#define FETL_DEFS_H_

#include "primitives/primitives.h"

namespace simpla
{

enum TopologyID
{
	INullForm = -1,

	IZeroForm = 0, IOneForm = 1, ITwoForm = 2, IThreeForm = 3, IFourForm = 4
}
;

//template<typename TL, typename TR>
//struct FieldEquation
//{
//public:
//
//	typename TypeTraits<TL>::ConstReference lhs_;
//	typename TypeTraits<TR>::ConstReference rhs_;
//
//	FieldEquation(TL const & lhs, TR const & rhs) :
//			lhs_(lhs), rhs_(rhs)
//	{
//	}
//
//};
//
//template<typename TG, int IFORM, typename TLExpr, typename TRExpr>
//FieldEquation<Field<TG, IFORM, TLExpr>, Field<TG, IFORM, TRExpr> >            //
//operator==(Field<TG, IFORM, TLExpr> const & lhs,
//		Field<TG, IFORM, TRExpr> const & rhs)
//{
//	return (FieldEquation<Field<TG, IFORM, TLExpr>, Field<TG, IFORM, TRExpr> >(
//			lhs, rhs));
//}

//Default fields are real value

template<typename TV> class Field;

#define DEFINE_FIELDS(TG)                                                         \
typedef TG Grid;                                                                     \
typedef Field<Grid, IZeroForm, Real>     ZeroForm;                                \
typedef Field<Grid, IOneForm, Real>      OneForm;                                  \
typedef Field<Grid, ITwoForm, Real>      TwoForm;                                  \
typedef Field<Grid, IThreeForm, Real>    ThreeForm;                              \
                                                                                     \
typedef Field<Grid, IZeroForm, nTuple<THREE,Real> >   VecZeroForm;                           \
typedef Field<Grid, IOneForm, nTuple<THREE,Real> >    VecOneForm;                             \
typedef Field<Grid, ITwoForm, nTuple<THREE,Real> >    VecTwoForm;                             \
typedef Field<Grid, IThreeForm, nTuple<THREE,Real> >  VecThreeForm;                         \
                                                                                     \
typedef Field<Grid, IZeroForm, Real> ScalarField;                             \
typedef Field<Grid, IZeroForm, nTuple<THREE,Real> > VecField;                              \
                                                                                     \
typedef Field<Grid, IZeroForm, Real>     RZeroForm;                               \
typedef Field<Grid, IOneForm, Real>      ROneForm;                                 \
typedef Field<Grid, ITwoForm, Real>      RTwoForm;                                 \
typedef Field<Grid, IThreeForm, Real>    RThreeForm;                             \
                                                                                     \
typedef Field<Grid, IZeroForm, nTuple<THREE,Real> >  RVecZeroForm;                         \
typedef Field<Grid, IOneForm, nTuple<THREE,Real> >   RVecOneForm;                           \
typedef Field<Grid, ITwoForm, nTuple<THREE,Real> >   RVecTwoForm;                           \
typedef Field<Grid, IThreeForm, nTuple<THREE,Real> > RVecThreeForm;                       \
                                                                                     \
typedef Field<Grid, IZeroForm, Real> RScalarField;                            \
typedef Field<Grid, IZeroForm, nTuple<THREE,Real> > RVecField;                            \
                                                                                     \
typedef Field<Grid, IZeroForm, Complex> CZeroForm;                          \
typedef Field<Grid, IOneForm, Complex>  COneForm;                            \
typedef Field<Grid, ITwoForm, Complex>  CTwoForm;                            \
typedef Field<Grid, IThreeForm, Complex>  CThreeForm;                       \
	                                                                         \
typedef Field<Grid, IZeroForm, nTuple<THREE,Complex> >   CVecZeroForm;                         \
typedef Field<Grid, IZeroForm, nTuple<THREE,Complex> >   CVecOneForm;                         \
typedef Field<Grid, IZeroForm, nTuple<THREE,Complex> >   CVecTwoForm;                         \
typedef Field<Grid, IThreeForm, nTuple<THREE,Complex> >  CVecThreeForm;                       \
	                                                                         \
typedef Field<Grid, IZeroForm, Complex> CScalarField;                       \
typedef Field<Grid, IZeroForm, nTuple<THREE,Complex> > CVecField;

} // namespace simpla

#endif /* FETL_DEFS_H_ */
