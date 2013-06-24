/*
 * fluid.h
 *
 *  Created on: 2011-12-10
 *      Author: salmon
 */

#ifndef FLUID_H_
#define FLUID_H_

#include "include/simpla_defs.h"
#include "engine/context.h"

#include "cold_fluid.h"
namespace simpla
{
namespace fliud
{

template<typename TG>
void RegisterModules(Context<TG> * ctx)
{
	DEFINE_FIELDS( TG)

	ctx->moduleFactory_["ColdFluid"] = TR1::bind(&ColdFluid<TG>::Create, ctx,
			TR1::placeholders::_1);

}

} //namespace fliud
} //namespace simpla

#endif /* FLUID_H_ */
