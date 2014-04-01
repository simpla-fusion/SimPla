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

template<typename TE, typename TB, typename TJ, typename TDict, typename TM, typename ...Args> optional<
        ParticleWrap<TE, TB, TJ>> CreateParticle(TDict const & dict, TM const & mesh, Args const & ...args)
{

	typedef TM Mesh;

	return Particle<PICEngineDeltaF<Mesh>>::template CreateWrap<TE, TB, TJ>(dict, mesh,
	        std::forward<Args const &>(args)...)
	        || Particle<PICEngineFull<Mesh>>::template CreateWrap<TE, TB, TJ>(dict, mesh,
	                std::forward<Args const &>(args)...)
	        || Particle<PICEngineGGauge<Mesh, Real, 4>>::template CreateWrap<TE, TB, TJ>(dict, mesh,
	                std::forward<Args const &>(args)...)
	        || Particle<PICEngineGGauge<Mesh, Real, 16>>::template CreateWrap<TE, TB, TJ>(dict, mesh,
	                std::forward<Args const &>(args)...)
	        || Particle<PICEngineGGauge<Mesh, Real, 32>>::template CreateWrap<TE, TB, TJ>(dict, mesh,
	                std::forward<Args const &>(args)...);

}

}  // namespace simpla

