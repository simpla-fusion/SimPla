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

#include "fetl_defs.h"

#include "ntuple.h"
#include "fields.h"
#include "arithmetic.h"
#include "vector_calculus.h"

namespace simpla
{
namespace fetl
{

//Default fields are real value

template<typename Grid, int IFORM, typename TV> struct Field;

#define DEFINE_FIELDS(TV,TG)                                                         \
typedef TG Grid;                                                                     \
typedef Field<Grid, IZeroForm, TV> ZeroForm;                                \
typedef Field<Grid, IOneForm, TV> OneForm;                                  \
typedef Field<Grid, ITwoForm, TV> TwoForm;                                  \
typedef Field<Grid, IThreeForm, TV> ThreeForm;                              \
                                                                                     \
typedef Field<Grid, IZeroForm, Vec3> VecZeroForm;                           \
typedef Field<Grid, IOneForm, Vec3> VecOneForm;                             \
typedef Field<Grid, ITwoForm, Vec3> VecTwoForm;                             \
typedef Field<Grid, IThreeForm, Vec3> VecThreeForm;                         \
                                                                                     \
typedef Field<Grid, IZeroForm, TV> ScalarField;                             \
typedef Field<Grid, IZeroForm, Vec3> VecField;                              \
                                                                                     \
typedef Field<Grid, IZeroForm, TV> RZeroForm;                               \
typedef Field<Grid, IOneForm, TV> ROneForm;                                 \
typedef Field<Grid, ITwoForm, TV> RTwoForm;                                 \
typedef Field<Grid, IThreeForm, TV> RThreeForm;                             \
                                                                                     \
typedef Field<Grid, IZeroForm, RVec3> RVecZeroForm;                         \
typedef Field<Grid, IOneForm, RVec3> RVecOneForm;                           \
typedef Field<Grid, ITwoForm, RVec3> RVecTwoForm;                           \
typedef Field<Grid, IThreeForm, RVec3> RVecThreeForm;                       \
                                                                                     \
typedef Field<Grid, IZeroForm, TV> RScalarField;                            \
typedef Field<Grid, IZeroForm, RVec3> RVecField;                            \
                                                                                     \
typedef Field<Grid, IZeroForm, Complex> CZeroForm;                          \
typedef Field<Grid, IOneForm, Complex> COneForm;                            \
typedef Field<Grid, ITwoForm, Complex> CTwoForm;                            \
typedef Field<Grid, IZeroForm, CVec3> CVecZeroForm;                         \
typedef Field<Grid, IZeroForm, Complex> CScalarField;                       \
typedef Field<Grid, IZeroForm, CVec3> CVecField;

}  // namespace fetl
} //namespace simpla

#endif  // FETL_H_
