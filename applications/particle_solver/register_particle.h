/*
 * register_particle.h
 *
 *  Created on: 2014年6月13日
 *      Author: salmon
 */

#ifndef REGISTER_PARTICLE_H_
#define REGISTER_PARTICLE_H_

#include <memory>
#include <string>

#include "../../src/utilities/primitives.h"
#include "../../src/particle/particle.h"
#include "../../src/particle/particle_base.h"
#include "../../src/particle/particle_factory.h"
#include "fluid_cold_engine.h"
#include "pic_engine_default.h"
#include "pic_engine_deltaf.h"
#include "pic_engine_ggauge.h"

namespace simpla
{

template<typename TP, typename ...Args> bool RegistOneParticle()
{
	std::function<std::shared_ptr<ParticleBase<typename TP::mesh_type>>(typename TP::mesh_type const &, Args &&...)> callback =
	        [](typename TP::mesh_type const &m, Args && ...args)
	        {
		        return CreateParticleWrap<TP, Args...>(m, std::forward<Args >(args)...);
	        };

	return RegisterParticle(TP::GetTypeAsString(), callback);
}

template<typename Mesh, typename ...Args>
bool RegisterAllParticles()
{

	bool res = true;

	res &= RegistOneParticle<Particle<PICEngineDefault<Mesh, true>>, Args...>();

	res &= RegistOneParticle<Particle<PICEngineDefault<Mesh, true>>, Args...>();

	res &= RegistOneParticle<Particle<PICEngineDefault<Mesh, false>>, Args...>();

	res &= RegistOneParticle<Particle<PICEngineDeltaF<Mesh>>, Args...>();

	res &= RegistOneParticle<Particle<PICEngineGGauge<Mesh, 4, true>>, Args...>();

	res &= RegistOneParticle<Particle<PICEngineGGauge<Mesh, 16, true>>, Args...>();

	res &= RegistOneParticle<Particle<PICEngineGGauge<Mesh, 32, true>>, Args...>();

	res &= RegistOneParticle<Particle<ColdFluid<Mesh> >, Args...>();

	return res;
}

}
// namespace simpla

#endif /* REGISTER_PARTICLE_H_ */
