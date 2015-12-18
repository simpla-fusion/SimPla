/*
 * register_Particle.h
 *
 * @date  2014-6-13
 *      @author  salmon
 */

#ifndef REGISTER_PARTICLE_H_
#define REGISTER_PARTICLE_H_

#include <string>

#include "../../core/particle/particle_base.h"
#include "../../core/utilities/factory.h"

namespace simpla
{

/**
 *  @ingroup particle
 *  @{  \defgroup  ParticleEngine particle Engine
 *  @}
 */

template<typename Mesh, typename ...Args>
Factory<std::string, ParticleBase, Args ...> RegisterAllParticles()
{

	Factory<std::string, ParticleBase, Args ...> factory;

//	factory.Register(Particle<mesh, ColdFluid>::template CreateFactoryFun<Args...>());

//	factory.Register(Particle<mesh, PICEngineFullF>::template CreateFactoryFun<Args...>());
//	factory.Register(Particle<mesh, PICDeltaF>::template CreateFactoryFun<Args...>());
//	factory.Register(Particle<PICEngineImplicit<mesh>>::template CreateFactoryFun<Args...>());
//	factory.Register(Particle<PICEngineGGauge<mesh, 4, true>>::template CreateFactoryFun<Args...>());
//	factory.Register(Particle<PICEngineGGauge<mesh, 16, true>>::template CreateFactoryFun<Args...>());
//	factory.Register(Particle<PICEngineGGauge<mesh, 32, true>>::template CreateFactoryFun<Args...>());

	return std::move(factory);
}

}
// namespace simpla

#endif /* REGISTER_PARTICLE_H_ */
