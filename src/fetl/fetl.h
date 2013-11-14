/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id: FETL.h 1009 2011-02-07 23:20:45Z salmon $
 * FETL.h
 *
 *  Created on: 2009-3-31
 *      Author: salmon
 */
#ifndef FETL_H_
#define FETL_H_

#include "./primitives.h"
#include "./ntuple.h"
#include "./field.h"
#include "./ntuple.h"
#include "./geometry.h"
#include "./field_ops.h"

#ifndef NO_FIELD_IO_CACHE
#  include "./field_rw_cache.h"
#endif
//namespace simpla
//{
//
//#define DEFINE_FIELDS(TG)   using namespace TG##Define;                      \
//
//
//typedef TG Mesh;                                                                     \
//typedef Field<Geometry<Mesh,0>,Real >     ZeroForm;                                \
//typedef Field<Geometry<Mesh,1>,Real >      OneForm;                                  \
//typedef Field<Geometry<Mesh,2>,Real >      TwoForm;                                  \
//typedef Field<Geometry<Mesh,3>,Real >    ThreeForm;                              \
//                                                                                        \
//typedef Field<Geometry<Mesh,0>, nTuple<3,Real> >   VecZeroForm;                           \
//typedef Field<Geometry<Mesh,1>, nTuple<3,Real> >    VecOneForm;                             \
//typedef Field<Geometry<Mesh,2>, nTuple<3,Real> >    VecTwoForm;                             \
//typedef Field<Geometry<Mesh,3>, nTuple<3,Real> >  VecThreeForm;                         \
//                                                                                     \
//typedef Field<Geometry<Mesh,0>, Real > ScalarField;                             \
//typedef Field<Geometry<Mesh,0>, nTuple<3,Real> > VecField;                              \
//                                                                                     \
//typedef Field<Geometry<Mesh,0>,Real >     RZeroForm;                               \
//typedef Field<Geometry<Mesh,1>,Real >      ROneForm;                                 \
//typedef Field<Geometry<Mesh,2>,Real >      RTwoForm;                                 \
//typedef Field<Geometry<Mesh,3>,Real >    RThreeForm;                             \
//                                                                                     \
//typedef Field<Geometry<Mesh,0>, nTuple<3,Real> >  RVecZeroForm;                         \
//typedef Field<Geometry<Mesh,1>, nTuple<3,Real> >   RVecOneForm;                           \
//typedef Field<Geometry<Mesh,2>, nTuple<3,Real> >   RVecTwoForm;                           \
//typedef Field<Geometry<Mesh,3>, nTuple<3,Real> > RVecThreeForm;                       \
//                                                                                     \
//typedef Field<Geometry<Mesh,0>, Real > RScalarField;                            \
//typedef Field<Geometry<Mesh,0>, nTuple<3,Real> > RVecField;                            \
//                                                                                     \
//typedef Field<Geometry<Mesh,0>, Complex > CZeroForm;                          \
//typedef Field<Geometry<Mesh,1>, Complex >  COneForm;                            \
//typedef Field<Geometry<Mesh,2>, Complex >  CTwoForm;                            \
//typedef Field<Geometry<Mesh,3>, Complex >  CThreeForm;                       \
//	                                                                         \
//typedef Field<Geometry<Mesh,0>, nTuple<3,Complex> >   CVecZeroForm;                         \
//typedef Field<Geometry<Mesh,0>, nTuple<3,Complex> >   CVecOneForm;                         \
//typedef Field<Geometry<Mesh,0>, nTuple<3,Complex> >   CVecTwoForm;                         \
//typedef Field<Geometry<Mesh,3>, nTuple<3,Complex> >  CVecThreeForm;                       \
//	                                                                         \
//typedef Field<Geometry<Mesh,0>, Complex > CScalarField;                       \
//typedef Field<Geometry<Mesh,0>, nTuple<3,Complex> > CVecField;                         \
//template<int IFORM, typename T> using Form = Field<Geometry<Mesh,IFORM>,T >;
//
//}// namespace simpla
#endif  // FETL_H_
