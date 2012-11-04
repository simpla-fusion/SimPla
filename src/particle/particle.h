/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_
#include "include/simpla_defs.h"
#include "engine/context.h"

namespace simpla
{

template<typename TG>
void RegisterParticles(Context<TG> * ctx)
{
//	ctx->objFactory_["DeltaF"] = TR1::bind(
//			&Context<TG>::template CreateObject<ParticleObject<TG, TS> >, ctx);
}

}  // namespace simpla

#endif /* PARTICLE_H_ */
