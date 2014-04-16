/*
 * particle_factory.h
 *
 *  Created on: 2014年2月11日
 *      Author: salmon
 */
#include "../../src/particle/particle.h"

#include "fluid_cold_engine.h"
#include "pic_engine_deltaf.h"
#include "pic_engine_full.h"
#include "pic_engine_ggauge.h"
namespace simpla
{

template<typename TParticle, typename ...Args>
std::shared_ptr<typename TParticle::base_type> CreateParticle_(std::string const & type_str, Args const & ... args)
{
	std::shared_ptr<typename TParticle::base_type> res(nullptr);

	if (type_str == TParticle::GetTypeAsString())
		res = std::dynamic_pointer_cast<typename TParticle::base_type>(
		        std::shared_ptr<typename TParticle::base_type>(new TParticle(std::forward<Args const &>(args)...)));

	return res;
}
template<typename Mesh, typename ...Args>
std::shared_ptr<ParticleBase<Mesh>> CreateParticle(Args const & ...args)
{
	std::shared_ptr<ParticleBase<Mesh>> res(nullptr);

	if (res == nullptr)
		res = CreateParticle_<Particle<PICEngineDeltaF<Mesh>>>(std::forward<Args const &>(args)...);

	if (res == nullptr)
		res = CreateParticle_<Particle<PICEngineFull<Mesh>>>(std::forward<Args const &>(args)...);

	if (res == nullptr)
		res = CreateParticle_<Particle<PICEngineGGauge<Mesh, Real, 4>>>(std::forward<Args const &>(args)...);

	if (res == nullptr)
		res = CreateParticle_<Particle<PICEngineGGauge<Mesh, Real, 16>>>(std::forward<Args const &>(args)...);

	if (res == nullptr)
		res = CreateParticle_<Particle<PICEngineGGauge<Mesh, Real, 32>>>(std::forward<Args const &>(args)...);

	if (res == nullptr)
		res = CreateParticle_<Particle<ColdFluid<Mesh>>>(std::forward<Args const &>(args)...);

	return res;
}

}
// namespace simpla

