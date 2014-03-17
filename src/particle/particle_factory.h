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

template<typename TE, typename TB, typename TJ>
struct ParticleWrap
{
	std::function<void(Real dt, TE const & E, TB const & B)> NextTimeStep;

	std::function<void(TJ * J, TE const & E, TB const & B)> Scatter;

	std::function<std::ostream &(std::ostream &)> Save;

	std::function<void(std::string const &)> DumpData;

};
template<typename TEngine, typename TE, typename TB, typename TJ, typename ...Args> bool _CreateParticle(
		typename TEngine::mesh_type const & mesh, std::string const& engine_type_str, ParticleWrap<TE, TB, TJ>* res,
		Args const & ...args)
{

	if (engine_type_str != TEngine::TypeName())
		return false;

	auto solver = std::shared_ptr<Particle<TEngine> >(new Particle<TEngine>(mesh));

	solver->Load(std::forward<Args const &>(args)...);

	using namespace std::placeholders;
	res->NextTimeStep = std::bind(&Particle<TEngine>::template NextTimeStep<TE, TB>, solver, _1, _2, _3);
	res->Scatter = std::bind(&Particle<TEngine>::template Scatter<TJ, TE, TB>, solver, _1, _2, _3);
	res->Save = std::bind(&Particle<TEngine>::Save, solver, _1);
	res->DumpData = std::bind(&Particle<TEngine>::DumpData, solver, _1);
	return true;
}

template<typename TM, typename TE, typename TB, typename TJ, typename ...Args> bool CreateParticle(TM const & mesh,
		std::string const& engine_type_str, ParticleWrap<TE, TB, TJ> * p, Args const & ...args)
{
	ParticleWrap<TE, TB, TJ> res;

	typedef TM Mesh;

	return _CreateParticle<PICEngineDeltaF<Mesh>, TE, TB, TJ>(mesh, engine_type_str, &res,
			std::forward<Args const &>(args)...)
			|| _CreateParticle<PICEngineFull<Mesh>, TE, TB, TJ>(mesh, engine_type_str, &res,
					std::forward<Args const &>(args)...)
			|| _CreateParticle<PICEngineGGauge<Mesh, 4>, TE, TB, TJ>(mesh, engine_type_str, &res,
					std::forward<Args const &>(args)...)
			|| _CreateParticle<PICEngineGGauge<Mesh, 16>, TE, TB, TJ>(mesh, engine_type_str, &res,
					std::forward<Args const &>(args)...)
			|| _CreateParticle<PICEngineGGauge<Mesh, 32>, TE, TB, TJ>(mesh, engine_type_str, &res,
					std::forward<Args const &>(args)...);

}

}  // namespace simpla

