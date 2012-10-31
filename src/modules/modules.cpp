/*
 * modules.cpp
 *
 *  Created on: 2012-10-31
 *      Author: salmon
 */

#include "modules.h"
namespace simpla
{

void RegisterModules(BaseContext * ctx)
{
	ctx->moduleFactory_["Loop"] = TR1::bind(&flow_control::Loop::Create, ctx,
			TR1::placeholders::_1);
	ctx->moduleFactory_["Clock"] = TR1::bind(&flow_control::Clock::Create, ctx,
			TR1::placeholders::_1);
	ctx->moduleFactory_["Field"] = TR1::bind(&preprocess::LoadField::Create,
			ctx, TR1::placeholders::_1);
	ctx->moduleFactory_["Particle"] = TR1::bind(
			&preprocess::LoadParticle::Create, ctx, TR1::placeholders::_1);

}

}  // namespace simpla
