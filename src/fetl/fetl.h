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

//Default fields are real value

template<typename Grid, int IFORM, typename TV> struct Field;

#define DEFINE_FIELDS(TV,TG)                                                         \
typedef TG Grid;                                                                     \
typedef Field<Grid, IZeroForm, TV>     ZeroForm;                                \
typedef Field<Grid, IOneForm, TV>      OneForm;                                  \
typedef Field<Grid, ITwoForm, TV>      TwoForm;                                  \
typedef Field<Grid, IThreeForm, TV>    ThreeForm;                              \
                                                                                     \
typedef Field<Grid, IZeroForm, Vec3>   VecZeroForm;                           \
typedef Field<Grid, IOneForm, Vec3>    VecOneForm;                             \
typedef Field<Grid, ITwoForm, Vec3>    VecTwoForm;                             \
typedef Field<Grid, IThreeForm, Vec3>  VecThreeForm;                         \
                                                                                     \
typedef Field<Grid, IZeroForm, TV> ScalarField;                             \
typedef Field<Grid, IZeroForm, Vec3> VecField;                              \
                                                                                     \
typedef Field<Grid, IZeroForm, TV>     RZeroForm;                               \
typedef Field<Grid, IOneForm, TV>      ROneForm;                                 \
typedef Field<Grid, ITwoForm, TV>      RTwoForm;                                 \
typedef Field<Grid, IThreeForm, TV>    RThreeForm;                             \
                                                                                     \
typedef Field<Grid, IZeroForm, RVec3>  RVecZeroForm;                         \
typedef Field<Grid, IOneForm, RVec3>   RVecOneForm;                           \
typedef Field<Grid, ITwoForm, RVec3>   RVecTwoForm;                           \
typedef Field<Grid, IThreeForm, RVec3> RVecThreeForm;                       \
                                                                                     \
typedef Field<Grid, IZeroForm, TV> RScalarField;                            \
typedef Field<Grid, IZeroForm, RVec3> RVecField;                            \
                                                                                     \
typedef Field<Grid, IZeroForm, Complex> CZeroForm;                          \
typedef Field<Grid, IOneForm, Complex>  COneForm;                            \
typedef Field<Grid, ITwoForm, Complex>  CTwoForm;                            \
typedef Field<Grid, IThreeForm, Complex>  CThreeForm;                       \
	                                                                         \
typedef Field<Grid, IZeroForm, CVec3>   CVecZeroForm;                         \
typedef Field<Grid, IZeroForm, CVec3>   CVecOneForm;                         \
typedef Field<Grid, IZeroForm, CVec3>   CVecTwoForm;                         \
typedef Field<Grid, IThreeForm, CVec3>  CVecThreeForm;                       \
	                                                                         \
typedef Field<Grid, IZeroForm, Complex> CScalarField;                       \
typedef Field<Grid, IZeroForm, CVec3> CVecField;

template<typename > struct Context;

template<typename TG> void RegisterFields(Context<TG> * ctx)
{
	DEFINE_FIELDS(typename TG::ValueType, TG)

	ctx->objFactory_["ZeroForm"] = TR1::bind(
			&Context<TG>::template CreateObject<ZeroForm>, ctx);

	ctx->objFactory_["OneForm"] = TR1::bind(
			&Context<TG>::template CreateObject<OneForm>, ctx);

	ctx->objFactory_["TwoForm"] = TR1::bind(
			&Context<TG>::template CreateObject<TwoForm>, ctx);

	ctx->objFactory_["ThreeForm"] = TR1::bind(
			&Context<TG>::template CreateObject<ThreeForm>, ctx);

	ctx->objFactory_["VecZeroForm"] = TR1::bind(
			&Context<TG>::template CreateObject<VecZeroForm>, ctx);

	ctx->objFactory_["VecOneForm"] = TR1::bind(
			&Context<TG>::template CreateObject<VecOneForm>, ctx);

	ctx->objFactory_["VecTwoForm"] = TR1::bind(
			&Context<TG>::template CreateObject<VecTwoForm>, ctx);

	ctx->objFactory_["VecThreeForm"] = TR1::bind(
			&Context<TG>::template CreateObject<VecThreeForm>, ctx);

	ctx->objFactory_["CZeroForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CZeroForm>, ctx);

	ctx->objFactory_["COneForm"] = TR1::bind(
			&Context<TG>::template CreateObject<COneForm>, ctx);

	ctx->objFactory_["CTwoForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CTwoForm>, ctx);

	ctx->objFactory_["CThreeForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CThreeForm>, ctx);

	ctx->objFactory_["CVecZeroForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CVecZeroForm>, ctx);

	ctx->objFactory_["CVecOneForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CVecOneForm>, ctx);

	ctx->objFactory_["CTwoForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CTwoForm>, ctx);

	ctx->objFactory_["CVecThreeForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CVecThreeForm>, ctx);

}
} //namespace simpla

#endif  // FETL_H_
