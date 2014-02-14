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

	std::function<void(TJ * J, TE const & E, TB const & B)> Collect;

	std::function<std::ostream &(std::ostream &)> Save;

	std::function<void(std::string const &)> DumpData;

};
template<typename TEngine, typename TE, typename TB, typename TJ> bool _CreateParticle(
        typename TEngine::mesh_type const & mesh, LuaObject const& cfg, ParticleWrap<TE, TB, TJ>* res)
{

	if (cfg["Type"].as<std::string>() != TEngine::TypeName())
		return false;

	auto solver = std::shared_ptr<Particle<TEngine> >(new Particle<TEngine>(mesh));

	solver->Load(cfg);
	using namespace std::placeholders;
	res->NextTimeStep = std::bind(&Particle<TEngine>::template NextTimeStep<TE, TB>, solver, _1, _2, _3);
	res->Collect = std::bind(&Particle<TEngine>::template Collect<TJ, TE, TB>, solver, _1, _2, _3);
	res->Save = std::bind(&Particle<TEngine>::Save, solver, _1);
	res->DumpData = std::bind(&Particle<TEngine>::DumpData, solver, _1);
	return true;
}

template<typename TM, typename TE, typename TB, typename TJ> bool CreateParticle(TM const & mesh, LuaObject const& cfg,
        ParticleWrap<TE, TB, TJ> * p)
{
	ParticleWrap<TE, TB, TJ> res;

	typedef TM Mesh;

	return _CreateParticle<PICEngineDeltaF<Mesh>, TE, TB, TJ>(mesh, cfg, &res)
	        || _CreateParticle<PICEngineFull<Mesh>, TE, TB, TJ>(mesh, cfg, &res)
	        || _CreateParticle<PICEngineGGauge<Mesh, 4>, TE, TB, TJ>(mesh, cfg, &res)
	        || _CreateParticle<PICEngineGGauge<Mesh, 16>, TE, TB, TJ>(mesh, cfg, &res)
	        || _CreateParticle<PICEngineGGauge<Mesh, 32>, TE, TB, TJ>(mesh, cfg, &res);

}

}  // namespace simpla

