/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <exception>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../fetl/fetl.h"
#include "../fetl/save_field.h"

#include "../utilities/log.h"
#include "../utilities/memory_pool.h"
#include "../utilities/sp_type_traits.h"

#include "../parallel/parallel.h"
#include "../model/model.h"

#include "particle_pool.h"
#include "load_particle.h"
#include "save_particle.h"

#include "particle_boundary.h"

namespace simpla
{

/** @defgroup Particle Particle
 *
 */

/**
 *  \brief Particle class
 */
template<class Engine>
class Particle: public Engine, public ParticlePool<typename Engine::mesh_type, typename Engine::Point_s>
{

public:
	static constexpr int IForm = VERTEX;

	typedef Engine engine_type;

	typedef ParticlePool<typename Engine::mesh_type, typename Engine::Point_s> storage_type;

	typedef Particle<engine_type> this_type;

	typedef typename engine_type::mesh_type mesh_type;

	typedef typename engine_type::Point_s particle_type;

	typedef typename engine_type::scalar_type scalar_type;

	typedef particle_type value_type;

	typedef typename mesh_type::iterator iterator;

	typedef typename mesh_type::coordinates_type coordinates_type;

public:
	mesh_type const & mesh;
	//***************************************************************************************************
	// Constructor
	Particle(mesh_type const & pmesh);

	template<typename TDict> Particle(TDict const & dict, mesh_type const & pmesh);

	// Destructor
	~Particle();

	template<typename TDict> void Load(TDict const & dict);

	template<typename ...Args>
	static std::shared_ptr<this_type> Create(Args && ... args)
	{
		return LoadParticle<this_type>(std::forward<Args>(args)...);
	}

	void AddConstraint(std::function<void()> const &foo)
	{
		constraint_.push_back(foo);
	}

	void ApplyConstraints()
	{
		for (auto & f : constraint_)
		{
			f();
		}
	}

	typename engine_type::n_type n;

	typename engine_type::J_type J;

	template<typename TE, typename TB>
	void NextTimeStepZero(TE const &E, TB const & B);

	template<typename TE, typename TB>
	void NextTimeStepHalf(TE const &E, TB const & B);

	template<typename TJ, typename ...Others>
	void Scatter(TJ *J, Others && ... args) const;

	std::string Save(std::string const & path, bool is_verbose = false) const;

	std::list<std::function<void()> > constraint_;
};
template<typename Engine>

Particle<Engine>::Particle(mesh_type const & pmesh)
		: engine_type(pmesh), storage_type(pmesh), mesh(pmesh), n(mesh), J(mesh)
{
}
template<typename Engine>
template<typename TDict>
Particle<Engine>::Particle(TDict const & dict, mesh_type const & pmesh)
		: Particle(pmesh)
{
	Load(dict);
}
template<typename Engine>
template<typename TDict>
void Particle<Engine>::Load(TDict const & dict)
{
	engine_type::Load(dict);

	storage_type::Load(dict["url"].template as<std::string>());

	J.clear();

	n.clear();
}

template<typename Engine>
Particle<Engine>::~Particle()
{
}

//*************************************************************************************************

template<typename Engine>
std::string Particle<Engine>::Save(std::string const & path, bool is_verbose) const
{
	std::stringstream os;

	GLOBAL_DATA_STREAM.OpenGroup(path );

	if (is_verbose)
	{
		GLOBAL_DATA_STREAM.DisableCompactStorable();
		os

		<< engine_type::Save(path)

		<< "\n, particles = " << storage_type::Save("particles")

		;

		os << "\n, n =" << simpla::Save( "n",n);

		os << "\n, J =" << simpla::Save( "J",J);

		GLOBAL_DATA_STREAM.EnableCompactStorable();
	}
	else
	{

		os << "\n, n =" << simpla::Save( "n",n);

		os << "\n, J =" << simpla::Save( "J",J);

	}

	return os.str();
}

template<typename Engine>
template<typename TE, typename TB>
void Particle<Engine>::NextTimeStepZero(TE const & E, TB const & B)
{

	LOGGER << "Push particles to zero step [ " << engine_type::GetTypeAsString() << std::boolalpha
	        << " , Enable Implicit =" << engine_type::EnableImplicit << " ]";

	storage_type::Sort();

	Real dt = mesh.GetDt();

	J.clear();

	for (auto & cell : *this)
	{
		//TODO add rw cache
		for (auto & p : cell.second)
		{
			this->engine_type::NextTimeStepZero(&p, dt, &J, E, B);
		}
	}

	UpdateGhosts(&J);

	LOGGER << DONE;
	LOG_CMD(n -= Diverge(MapTo<EDGE>(J)) * dt);

}

template<typename Engine>
template<typename TE, typename TB>
void Particle<Engine>::NextTimeStepHalf(TE const & E, TB const & B)
{

	LOGGER << "Push particles to half step[ " << engine_type::GetTypeAsString() << std::boolalpha
	        << " , Enable Implicit =" << engine_type::EnableImplicit << " ]";

	Real dt = mesh.GetDt();

	storage_type::Sort();

	for (auto & cell : *this)
	{
		//TODO add rw cache
		for (auto & p : cell.second)
		{
			this->engine_type::NextTimeStepHalf(&p, dt, E, B);
		}
	}

	storage_type::Sort();
	ApplyConstraints();
	storage_type::Sort();
	LOGGER << DONE;
}

template<typename Engine> template<typename TJ, typename ...Others>
void Particle<Engine>::Scatter(TJ *pJ, Others &&... args) const
{

	LOGGER << "Scatter particles   ";

	for (auto & cell : *this)
	{
		for (auto const& p : cell.second)
		{
			this->engine_type::Scatter(p, pJ, std::forward<Others> (args)...);
		}
	}

	UpdateGhosts(pJ);

	LOGGER << DONE;
}
//*************************************************************************************************
template<typename TX, typename TV, typename TE, typename TB> inline
void BorisMethod(Real dt, Real cmr, TE const & E, TB const &B, TX *x, TV *v)
{
// \cite   Birdsall(1991)   p.62
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

#endif /* PARTICLE_H_ */
