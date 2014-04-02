/*
 * particle_factory.h
 *
 *  Created on: 2014年2月11日
 *      Author: salmon
 */
#include "particle.h"
#include "pic_engine_deltaf.h"
#include "pic_engine_full.h"
#include "pic_engine_ggauge.h"
namespace simpla
{

template<typename TE, typename TB, typename TJ, typename TDict, typename TM, typename ...Args> bool CreateParticle(
        ParticleWrap<TE, TB, TJ> * res, TDict const & dict, TM const & mesh, Args const & ...args)
{

	typedef TM Mesh;

	return

	Particle<PICEngineDeltaF<Mesh>>::CreateWrap(res, dict, mesh, std::forward<Args const &>(args)...)

	|| Particle<PICEngineFull<Mesh>>::CreateWrap(res, dict, mesh, std::forward<Args const &>(args)...)

	|| Particle<PICEngineGGauge<Mesh, Real, 4>>::CreateWrap(res, dict, mesh, std::forward<Args const &>(args)...)

	|| Particle<PICEngineGGauge<Mesh, Real, 16>>::CreateWrap(res, dict, mesh, std::forward<Args const &>(args)...)

	|| Particle<PICEngineGGauge<Mesh, Real, 32>>::CreateWrap(res, dict, mesh, std::forward<Args const &>(args)...)

	;

}

}  // namespace simpla

