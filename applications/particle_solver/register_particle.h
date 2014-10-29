/*
 * register_particle.h
 *
 * \date  2014-6-13
 *      \author  salmon
 */

#ifndef REGISTER_PARTICLE_H_
#define REGISTER_PARTICLE_H_

#include <memory>
#include <string>

#include "../../core/utilities/primitives.h"
#include "../../core/particle/particle.h"
#include "../../core/particle/particle_base.h"
#include "fluid_cold_engine.h"
//#include "pic_engine_fullf.h"
#include "pic_engine_deltaf.h"
//#include "pic_engine_implicit.h"
//#include "pic_engine_ggauge.h"
namespace simpla
{

/**
 *  \ingroup Particle
 *  @{  \defgroup  ParticleEngine Particle Engine
 *  @}
 */

template<typename Mesh, typename ...Args>
Factory<std::string, ParticleBase, Args ...> RegisterAllParticles()
{

	Factory<std::string, ParticleBase, Args ...> factory;

	factory.Register(Particle<Mesh, ColdFluid>::template CreateFactoryFun<Args...>());

//	factory.Register(Particle<Mesh, PICEngineFullF>::template CreateFactoryFun<Args...>());
	factory.Register(Particle<Mesh, PICDeltaF>::template CreateFactoryFun<Args...>());
//	factory.Register(Particle<PICEngineImplicit<Mesh>>::template CreateFactoryFun<Args...>());
//	factory.Register(Particle<PICEngineGGauge<Mesh, 4, true>>::template CreateFactoryFun<Args...>());
//	factory.Register(Particle<PICEngineGGauge<Mesh, 16, true>>::template CreateFactoryFun<Args...>());
//	factory.Register(Particle<PICEngineGGauge<Mesh, 32, true>>::template CreateFactoryFun<Args...>());

	return std::move(factory);
}

}
// namespace simpla

#endif /* REGISTER_PARTICLE_H_ */
