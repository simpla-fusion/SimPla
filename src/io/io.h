/*
 * io.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef IO_H_
#define IO_H_

#include "include/simpla_defs.h"
#include "engine/context.h"

#include "write_xdmf.h"

namespace simpla
{
namespace io
{

template<typename TG>
void RegisterModules(Context<TG> * ctx)
{
	DEFINE_FIELDS(TG)

	ctx->moduleFactory_["WriteXDMF"] = TR1::bind(&WriteXDMF<TG>::Create,
			ctx, TR1::placeholders::_1);

}

} //namespace io
} //namespace simpla

#endif /* IO_H_ */
