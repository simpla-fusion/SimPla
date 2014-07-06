/*
 * register_particle.h
 *
 * \date  2014年6月13日
 *      \author  salmon
 */

#ifndef REGISTER_PARTICLE_H_
#define REGISTER_PARTICLE_H_

#include <memory>
#include <string>

#include "../../src/utilities/primitives.h"
#include "../../src/particle/particle.h"
#include "../../src/particle/particle_base.h"
#include "fluid_cold_engine.h"
#include "pic_engine_default.h"
#include "pic_engine_deltaf.h"
#include "pic_engine_ggauge.h"

namespace simpla
{

/**
 *  \ingroup Particle
 *  @{  @defgroup ParticleEngine Particle Engine
 *  @}
 */

template<typename Mesh, typename ...Args>
Factory<std::string, ParticleBase<Mesh>, Args ...> RegisterAllParticles()
{

	Factory<std::string, ParticleBase<Mesh>, Args ...> factory;

	ParticleWrap<Particle<PICEngineDefault<Mesh, true>>> ::Register(&factory);
	ParticleWrap<Particle<PICEngineDefault<Mesh, false>>> ::Register(&factory);
	ParticleWrap<Particle<PICEngineDeltaF<Mesh>>> ::Register(&factory);
	ParticleWrap<Particle<PICEngineGGauge<Mesh, 4, true>>> ::Register(&factory);
	ParticleWrap<Particle<PICEngineGGauge<Mesh, 16, true>>> ::Register(&factory);
	ParticleWrap<Particle<PICEngineGGauge<Mesh, 32, true>>> ::Register(&factory);
	ParticleWrap<Particle<ColdFluid<Mesh>>> ::Register(&factory);

	return std::move(factory);
}

}
// namespace simpla

#endif /* REGISTER_PARTICLE_H_ */
