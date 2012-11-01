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
#include "engine/detail/context_impl.h"

#include "modules/em/em.h"
#include "modules/fluid/fluid.h"
#include "modules/field_fun/field_fun.h"
#include "modules/io/io.h"
#include "modules/preprocess/preprocess.h"
#include "modules/flow_control/flow_control.h"
namespace simpla
{

template<typename TG>
void RegisterModules(Context<TG> * ctx)
{

	em::RegisterModules(ctx);
	io::RegisterModules(ctx);
	fliud::RegisterModules(ctx);
	field_fun::RegisterModules(ctx);
	flow_control::RegisterModules(ctx);
	preprocess::RegisterModules(ctx);

}

}  // namespace simpla

#endif /* MODULES_H_ */
