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

template<typename TG,typename TV> class Field;

#define DEFINE_FIELDS(TG)                                                         \
typedef TG Grid;                                                                     \
typedef Field<ZeroForm<Grid>, Real>     ZeroForm;                                \
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
