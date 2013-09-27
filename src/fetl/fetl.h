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

#include "field.h"
#include "expression.h"
#include "vector_calculus.h"

#include "fetl/primitives.h"
#include <vector>
namespace simpla
{

template<typename TG, typename TV> class Field;
template<typename, int> class Geometry;

#define DEFINE_FIELDS(TG)                                                         \
typedef TG Mesh;                                                                     \
typedef Field<Geometry<Mesh,0>,std::vector<Real> >     ZeroForm;                                \
typedef Field<Geometry<Mesh,1>,std::vector<Real> >      OneForm;                                  \
typedef Field<Geometry<Mesh,2>,std::vector<Real> >      TwoForm;                                  \
typedef Field<Geometry<Mesh,3>,std::vector<Real> >    ThreeForm;                              \
                                                                                        \
typedef Field<Geometry<Mesh,0>, std::vector<nTuple<3,Real> > >   VecZeroForm;                           \
typedef Field<Geometry<Mesh,1>, std::vector<nTuple<3,Real> > >    VecOneForm;                             \
typedef Field<Geometry<Mesh,2>, std::vector<nTuple<3,Real> > >    VecTwoForm;                             \
typedef Field<Geometry<Mesh,3>, std::vector<nTuple<3,Real> > >  VecThreeForm;                         \
                                                                                     \
typedef Field<Geometry<Mesh,0>, std::vector<Real> > ScalarField;                             \
typedef Field<Geometry<Mesh,0>, std::vector<nTuple<3,Real> > > VecField;                              \
                                                                                     \
typedef Field<Geometry<Mesh,0>,std::vector<Real> >     RZeroForm;                               \
typedef Field<Geometry<Mesh,1>,std::vector<Real> >      ROneForm;                                 \
typedef Field<Geometry<Mesh,2>,std::vector<Real> >      RTwoForm;                                 \
typedef Field<Geometry<Mesh,3>,std::vector<Real> >    RThreeForm;                             \
                                                                                     \
typedef Field<Geometry<Mesh,0>, std::vector<nTuple<3,Real> > >  RVecZeroForm;                         \
typedef Field<Geometry<Mesh,1>, std::vector<nTuple<3,Real> > >   RVecOneForm;                           \
typedef Field<Geometry<Mesh,2>, std::vector<nTuple<3,Real> > >   RVecTwoForm;                           \
typedef Field<Geometry<Mesh,3>, std::vector<nTuple<3,Real> > > RVecThreeForm;                       \
                                                                                     \
typedef Field<Geometry<Mesh,0>, std::vector<Real> > RScalarField;                            \
typedef Field<Geometry<Mesh,0>, std::vector<nTuple<3,Real> > > RVecField;                            \
                                                                                     \
typedef Field<Geometry<Mesh,0>, std::vector<Complex > > CZeroForm;                          \
typedef Field<Geometry<Mesh,1>, std::vector<Complex > >  COneForm;                            \
typedef Field<Geometry<Mesh,2>, std::vector<Complex > >  CTwoForm;                            \
typedef Field<Geometry<Mesh,3>, std::vector<Complex > >  CThreeForm;                       \
	                                                                         \
typedef Field<Geometry<Mesh,0>, std::vector<nTuple<3,Complex> > >   CVecZeroForm;                         \
typedef Field<Geometry<Mesh,0>, std::vector<nTuple<3,Complex> > >   CVecOneForm;                         \
typedef Field<Geometry<Mesh,0>, std::vector<nTuple<3,Complex> > >   CVecTwoForm;                         \
typedef Field<Geometry<Mesh,3>, std::vector<nTuple<3,Complex> > >  CVecThreeForm;                       \
	                                                                         \
typedef Field<Geometry<Mesh,0>, std::vector<Complex > > CScalarField;                       \
typedef Field<Geometry<Mesh,0>, std::vector<nTuple<3,Complex> > > CVecField;                         \
//template<int IFORM,typename T> using Form 	= Field<Geometry<Mesh,IFORM>,std::vector<T> >;        \



template<typename T> struct ContainerField;

template<typename TG, typename TE>
struct ContainerField<Field<TG, std::vector<TE> > >
{
	typedef Field<TG, TE> type;
};

}  // namespace simpla
#endif  // FETL_H_
