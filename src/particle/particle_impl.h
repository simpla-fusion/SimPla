/*
 * particle_impl.h
 *
 *  Created on: 2014年8月29日
 *      Author: salmon
 */

#ifndef PARTICLE_IMPL_H_
#define PARTICLE_IMPL_H_

#include <exception>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../utilities/log.h"
#include "../utilities/sp_type_traits.h"
#include "../utilities/container_pools.h"
#include "../parallel/parallel.h"
#include "../model/model.h"

#include "particle_base.h"
#include "particle_pool.h"
#include "load_particle.h"
#include "save_particle.h"

namespace simpla
{

/** \defgroup  Particle Particle
 *
 */

/**
 *  \brief Particle class
 *
 *  this class is a proxy between ParticleBase and Engine,ParticlePool
 */
template<typename TM, class Engine>
class Particle: public Engine, public ContainerPool<size_t, typename Engine::Point_s>
{

public:
	static constexpr unsigned int IForm = VERTEX;

	typedef TM mesh_type;

	typedef Engine engine_type;

	typedef typename engine_type::Point_s particle_type;

	typedef ContainerPool<size_t, particle_type> storage_type;

	typedef Particle<mesh_type, engine_type> this_type;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

public:
	mesh_type const & mesh;
	Properties properties;

	//***************************************************************************************************
	// Constructor
	template<typename TModel, typename ...Others>
	Particle(TModel const & pmesh, Others && ...);
	// Destructor
	~Particle();

	template<typename ...Args>
	static std::shared_ptr<ParticleBase> create(Args && ... args)
	{
		return LoadParticle<this_type>(std::forward<Args>(args)...);
	}

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others);

	std::string save(std::string const & path) const;

	std::ostream& print(std::ostream & os) const
	{
		os << ",";
		engine_type::properties.print(os);
		return os;
	}

	bool is_implicit() const
	{
		return engine_type::properties.get("IsImplicit", false);
	}

	template<typename ...Args>
	void next_timestep(Args && ...);

	template<typename TJ>
	void ScatterJ(TJ * J) const;

	template<typename TJ>
	void ScatterRho(TJ * rho) const;

};

template<typename TM, typename Engine>
template<typename TModel, typename ... Others>
Particle<TM, Engine>::Particle(TModel const & model, Others && ...others) :
		storage_type(model), mesh(model), rho(model), J(model)
{
	this_type::load(std::forward<Others>(others)...);
}

template<typename TM, typename Engine>
Particle<TM, Engine>::~Particle()
{
}
template<typename TM, typename Engine>
template<typename TDict, typename ...Others>
void Particle<TM, Engine>::load(TDict const & dict, Others && ...others)
{
	engine_type::load(dict, std::forward<Others>(others)...);

	storage_type::load(dict, std::forward<Others>(others)...);

	properties.set("DumpParticle", dict["DumpParticle"].template as<bool>(false));

	properties.set("DivergeJ", dict["DivergeJ"].template as<bool>(false));

	properties.set("ScatterN", dict["ScatterN"].template as<bool>(false));

	J.clear();

	rho.clear();

}
template<typename ...T>
std::ostream &operator<<(std::ostream&os, Particle<T...> const & p)
{
	p.print(os);
	return os;
}

//*************************************************************************************************

template<typename TM, typename Engine>
std::string Particle<TM, Engine>::save(std::string const & path) const
{
	std::stringstream os;

	GLOBAL_DATA_STREAM.cd(path);

	os << "\n, n =" << simpla::save("rho", rho);

	os << "\n, J =" << simpla::save("J", J);

	if (properties["DumpParticle"].template as<bool>(false))
	{
		os << "\n, particles = " << storage_type::save("particles");
	}

	return os.str();
}

template<typename TM, typename Engine>
template<typename ...Args>
void Particle<TM, Engine>::next_timestep(Args && ...args)
{

	LOGGER << "Push particles to half step [ " << engine_type::get_type_as_string() << " ]";

	storage_type::Sort();

	Real dt = mesh.get_dt();

	for (auto & cell : *this)
	{
		//TODO add rw cache
		for (auto & p : cell.second)
		{
			this->engine_type::next_timestep(&p, std::forward<Args>(args)...);
		}
	}

	storage_type::set_changed();
}

template<typename TM, typename Engine>
void Particle<TM, Engine>::update_fields()
{

	LOGGER
	articles to
	fields[" << engine_type::get_type_as_string() << "]
	";

	Real dt = mesh.get_dt();

	storage_type::Sort();

	J.clear();

	for (auto & cell : *this)
	{
		//TODO add rw cache
		for (auto & p : cell.second)
		{
			this->engine_type::ScatterJ(p, &J);
		}
	}

	update_ghosts(&J);

	if (engine_type::properties["DivergeJ"].template as<bool>(false))
	{
		LOG_CMD(rho -= Diverge(MapTo<EDGE>(J)) * dt);
	}
	else if (engine_type::properties["ScatterN"].template as<bool>(false))
	{
		rho.clear();

		for (auto & cell : *this)
		{
			//TODO add rw cache
			for (auto & p : cell.second)
			{
				this->engine_type::ScatterRho(p, &rho);
			}
		}

		update_ghosts(&rho);
	}
}

template<typename TM, typename Engine>
template<typename TRange, typename TFun>
void Particle<TM, Engine>::remove(TRange const & range, TFun const & obj)
{
	auto f1 = TypeCast<std::function<bool(coordinates_type const &, Vec3 const &)>>(obj);

	std::function<bool(particle_type const&)> fun = [&](particle_type const & p)
	{

		auto z=engine_type::pull_back(p);

		return f1(std::get<0>(z),std::get<1>(z));

	};

	storage_type::remove(range, fun);

}

template<typename TM, typename Engine>
template<typename TRange, typename TFun>
void Particle<TM, Engine>::modify(TRange const & range, TFun const & obj)
{

	auto f1 = TypeCast<std::function<std::tuple<coordinates_type, Vec3>(coordinates_type const &, Vec3 const &)>>(obj);

	std::function<void(particle_type *)> fun = [&](particle_type * p)
	{
		auto z0=engine_type::pull_back(*p);

		auto z1=f1(std::get<0>(z0),std::get<1>(z0));

		*p=engine_type::push_forward( std::get<0>(z1),std::get<1>(z1),std::get<2>(z0));
	};

	storage_type::Modify(range, fun);

}

//*************************************************************************************************
template<typename TX, typename TV, typename TE, typename TB> inline
void BorisMethod(Real dt, Real cmr, TE const & E, TB const &B, TX *x, TV *v)
{
// \note   Birdsall(1991)   p.62
// Bories Method

	Vec3 v_;

	auto t = B * (cmr * dt * 0.5);

	(*v) += E * (cmr * dt * 0.5);

	v_ = (*v) + Cross((*v), t);

	(*v) += Cross(v_, t) * (2.0 / (Dot(t, t) + 1.0));

	(*v) += E * (cmr * dt * 0.5);

	(*x) += (*v) * dt;

}

}
// namespace simpla

#endif /* PARTICLE_IMPL_H_ */
