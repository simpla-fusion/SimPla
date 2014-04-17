/*
 * em.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef EM_H_
#define EM_H_

#include "simpla_defs.h"
#include "engine/context.h"

#include "maxwell.h"
#include "pml.h"

namespace simpla
{
namespace em
{

template<typename TG>
void RegisterModules(Context<TG> * ctx)
{
	DEFINE_FIELDS( TG)

	ctx->moduleFactory_["Maxwell"] = TR1::bind(&Maxwell<TG>::Create, ctx,
			TR1::placeholders::_1);

	ctx->moduleFactory_["PML"] = TR1::bind(&PML<TG>::Create, ctx,
			TR1::placeholders::_1);
}

} //namespace em
} //namespace simpla

#endif /* EM_H_ */
