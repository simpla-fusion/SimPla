/*
 * preprocess.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PREPROCESS_H_
#define PREPROCESS_H_
#include "include/simpla_defs.h"
#include "engine/context.h"
#include "utilities/properties.h"

#include "loadfield.h"
#include "loadparticle.h"

namespace simpla
{
namespace preprocess
{

template<typename TCTX>
inline void RegisterModules(TCTX * ctx)
{

	ctx->moduleFactory_["Field"] = TR1::bind(&preprocess::LoadField::Create,
			ctx, TR1::placeholders::_1);

	ctx->moduleFactory_["Particle"] = TR1::bind(
			&preprocess::LoadParticle::Create, ctx, TR1::placeholders::_1);

}
}  // namespace preprocess
}  // namespace simpla

#endif /* PREPROCESS_H_ */
