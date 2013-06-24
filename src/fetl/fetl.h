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
#include "field.h"
#include "arithmetic.h"
#include "vector_calculus.h"

//namespace simpla
//{
//template<typename > struct Context;
//
//
//template<typename TG> void RegisterFields(Context<TG> * ctx)
//{
//	DEFINE_FIELDS( TG)
//
//	ctx->objFactory_["Field.ZeroForm"] = TR1::bind(&ZeroForm::Create, ctx,
//			TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.OneForm"] = TR1::bind(&OneForm::Create, ctx,
//			TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.TwoForm"] = TR1::bind(&TwoForm::Create, ctx,
//			TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.ThreeForm"] = TR1::bind(&ThreeForm::Create, ctx,
//			TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.VecZeroForm"] = TR1::bind(&VecZeroForm::Create, ctx,
//			TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.VecOneForm"] = TR1::bind(&VecOneForm::Create, ctx,
//			TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.VecTwoForm"] = TR1::bind(&VecTwoForm::Create, ctx,
//			TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.VecThreeForm"] = TR1::bind(&VecThreeForm::Create,
//			ctx, TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.CZeroForm"] = TR1::bind(&CZeroForm::Create, ctx,
//			TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.COneForm"] = TR1::bind(&COneForm::Create, ctx,
//			TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.CTwoForm"] = TR1::bind(&CTwoForm::Create, ctx,
//			TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.CThreeForm"] = TR1::bind(&CThreeForm::Create, ctx,
//			TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.CVecZeroForm"] = TR1::bind(&CVecZeroForm::Create,
//			ctx, TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.CVecOneForm"] = TR1::bind(&CVecOneForm::Create, ctx,
//			TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.CTwoForm"] = TR1::bind(&CTwoForm::Create, ctx,
//			TR1::placeholders::_1);
//
//	ctx->objFactory_["Field.CVecThreeForm"] = TR1::bind(&CVecThreeForm::Create,
//			ctx, TR1::placeholders::_1);
//
//}
//
//}  // namespace simpla
#endif  // FETL_H_
