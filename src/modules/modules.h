/*
 * modules.h
 *
 *  Created on: 2012-10-31
 *      Author: salmon
 */

#ifndef MODULES_H_
#define MODULES_H_
#include "include/simpla_defs.h"

#include "engine/context.h"

#include "modules/em/maxwell.h"
#include "modules/em/pml.h"
#include "modules/em/rf_src.h"
#include "modules/fluid/cold_fluid.h"

#include "modules/field_fun/field_fun.h"

#include "modules/preprocess/loadfield.h"
#include "modules/preprocess/loadparticle.h"

#include "modules/io/write_xdmf.h"
#include "modules/io/read_hdf5.h"

#include "modules/flow_control/flow_control.h"

namespace simpla
{
void RegisterModules(BaseContext * ctx);

template<typename TG>
void RegisterModules(Context<TG> * ctx)
{
	ctx->moduleFactory_["WriteXDMF"] = TR1::bind(&io::WriteXDMF<TG>::Create,
			ctx, TR1::placeholders::_1);

	ctx->moduleFactory_["RFSrc"] = TR1::bind(&em::RFSrc<TG>::Create, ctx,
			TR1::placeholders::_1);

	ctx->moduleFactory_["Maxwell"] = TR1::bind(&em::Maxwell<TG>::Create, ctx,
			TR1::placeholders::_1);

	ctx->moduleFactory_["PML"] = TR1::bind(&em::PML<TG>::Create, ctx,
			TR1::placeholders::_1);

	ctx->moduleFactory_["ColdFluid"] = TR1::bind(&em::ColdFluid<TG>::Create,
			ctx, TR1::placeholders::_1);

	ctx->moduleFactory_["RampWave"] = TR1::bind(
			field_fun::Create<TG, field_fun::RampWave>, ctx,
			TR1::placeholders::_1);

	//	moduleFactory_["Smooth"] = TR1::bind(
	//			&field_fun::Create<TG, field_fun::Smooth>, ctx,
	//			TR1::placeholders::_1);
	//
	//	moduleFactory_["Damping"] = TR1::bind(
	//			&field_fun::Create<TG, field_fun::Damping>, ctx,
	//			TR1::placeholders::_1);

}

}  // namespace simpla

#endif /* MODULES_H_ */
