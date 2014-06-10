/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <cstddef>
#include <exception>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../fetl/fetl.h"
#ifndef NO_RW_CACHE
#	include "../fetl/field_rw_cache.h"
#endif //NO_RW_CACHE
#include "../utilities/log.h"
#include "../utilities/memory_pool.h"
#include "../utilities/type_utilites.h"

#include "../parallel/parallel.h"

#include "../io/data_stream.h"

#include "../fetl/save_field.h"
#include "particle_pool.h"
#include "load_particle.h"
#include "save_particle.h"
#include "particle_boundary.h"

#include "../model/command.h"
#include "../model/material.h"

namespace simpla
{

//*******************************************************************************************************
template<class Engine>
class Particle: public Engine, public ParticlePool<typename Engine::mesh_type, typename Engine::Point_s>
{
	std::mutex write_lock_;

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

	template<typename ...Args> Particle(mesh_type const & pmesh, Args const & ...others);

	// Destructor
	virtual ~Particle();

	template<typename TDict, typename ...Args> void Load(TDict const & dict, Args const & ...others);

	void AddCommand(std::function<void()> const &foo)
	{
		commands_.push_back(foo);
	}

	//***************************************************************************************************
	// Interface

	typename engine_type::n_type n;

	typename engine_type::J_type J;

	template<typename TE, typename TB>
	void NextTimeStepZero(TE const &E, TB const & B);

	template<typename TE, typename TB>
	void NextTimeStepHalf(TE const &E, TB const & B);

	template<typename TJ, typename ...Args>
	void Scatter(TJ *J, Args const & ... args) const;

	std::string Save(std::string const & path, bool is_verbose = false) const;

private:

	template<typename TRange, typename ...Args>
	void Scatter(TRange const & range, Cache<Args> &&... args) const;

	template<typename TDict, typename ...Others>
	void AddCommand(TDict const & dict, Others const & ...others);

	std::list<std::function<void()> > commands_;
};
template<typename Engine>

Particle<Engine>::Particle(mesh_type const & pmesh)
		: engine_type(pmesh), storage_type(pmesh), mesh(pmesh), n(mesh), J(mesh)
{
}
template<typename Engine>
template<typename ...Others>
Particle<Engine>::Particle(mesh_type const & pmesh, Others const & ...others)
		: Particle(pmesh)
{
	Load(std::forward<Others const &>(others)...);
}
template<typename Engine>
template<typename TDict, typename ...Others>
void Particle<Engine>::Load(TDict const & dict, Others const & ...others)
{
	engine_type::Load(dict, std::forward<Others const &>(others)...);

	storage_type::Load(dict, std::forward<Others const &>(others)...);

	LoadParticle(this, dict, std::forward<Others const &>(others)...);

	AddCommand(dict["Commands"], std::forward<Others const &>(others)...);

}

template<typename Engine>
Particle<Engine>::~Particle()
{
}
template<typename Engine>
template<typename TDict, typename ...Others> void Particle<Engine>::AddCommand(TDict const & dict,
        Others const & ...others)
{
	if (!dict.is_table())
		return;
	for (auto item : dict)
	{
		auto dof = item.second["DOF"].template as<std::string>("");

		if (dof == "n")
		{

			LOGGER << "Add constraint to " << dof;

			commands_.push_back(
			        Command<typename engine_type::n_type>::Create(&n, item.second,
			                std::forward<Others const &>(others)...));

		}
		else if (dof == "J")
		{

			LOGGER << "Add constraint to " << dof;

			commands_.push_back(
			        Command<typename engine_type::J_type>::Create(&J, item.second,
			                std::forward<Others const &>(others)...));

		}
		else if (dof == "ParticlesBoundary")
		{

			LOGGER << "Add constraint to " << dof;

			commands_.push_back(
			        BoundaryCondition<this_type>::Create(this, item.second, std::forward<Others const &>(others)...));
		}
		else
		{
			UNIMPLEMENT2("Unknown DOF!");
		}
		LOGGER << DONE;
	}

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

	J.Clear();

	ParallelDo(

	[&](int t_num,int t_id)
	{
		Cache<TE const &> cE(E);
		Cache<TB const &> cB(B);
		Cache<typename engine_type::J_type*> cJ(&J);

		for(auto s: this->mesh.Select(IForm).Split(t_num,t_id))
		{
			RefreshCache(s,cE,cB,cJ);
			for (auto & p : this->at(s) )
			{
				this->engine_type::NextTimeStepZero(&p,dt , (*cJ),*cE,*cB);
			}
			FlushCache(s,cJ);
		}

	});

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

	ParallelDo(

	[&](int t_num,int t_id)
	{

		Cache<TE const &> cE(E);
		Cache<TB const &> cB(B);

		for(auto s: this->mesh.Select(IForm).Split(t_num,t_id))
		{
			RefreshCache(s,cE,cB);
			for (auto & p : this->at(s) )
			{
				this->engine_type::NextTimeStepHalf(&p,dt ,*cE,*cB);
			}
		}

	});

	storage_type::Sort();

	LOGGER << DONE;
}
template<typename Engine>
template<typename TRange, typename ...Args>
void Particle<Engine>::Scatter(TRange const & range, Cache<Args> &&... args) const
{

	for (auto s : range)
	{
		RefreshCache(s, args...);
		for (auto const& p : this->at(s))
		{
			this->engine_type::Scatter(p, (*args)...);
		}
		FlushCache(s, args...);
	}

}
template<typename Engine> template<typename TJ, typename ...Args>
void Particle<Engine>::Scatter(TJ *pJ, Args const &... args) const
{
	ParallelDo(

	[&](int t_num,int t_id)
	{
		Scatter(this->mesh.Select(IForm).Split(t_num, t_id), Cache<TJ*> (pJ),Cache<Args const &>(args)...);
	});

	UpdateGhosts(pJ);
}
//*************************************************************************************************
template<typename TX, typename TV, typename TE, typename TB> inline
void BorisMethod(Real dt, Real cmr, TE const & E, TB const &B, TX *x, TV *v)
{
// @ref  Birdsall(1991)   p.62
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
