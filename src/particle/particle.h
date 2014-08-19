/*
 * particle.h
 *
 *  created on: 2012-11-1
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
class Particle: public ParticleBase, public Engine, public ParticlePool<TM, typename Engine::Point_s>
{

public:
	static constexpr unsigned int IForm = VERTEX;

	typedef TM mesh_type;

	typedef Engine engine_type;

	typedef ParticlePool<mesh_type, typename Engine::Point_s> storage_type;

	typedef Particle<mesh_type, engine_type> this_type;

	typedef typename engine_type::Point_s particle_type;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef particle_type value_type;

	typedef typename mesh_type::iterator iterator;

	typedef typename mesh_type::coordinates_type coordinates_type;

public:
	mesh_type const & mesh;
	//***************************************************************************************************
	// Constructor
	Particle(mesh_type const & pmesh);

	template<typename TDict, typename TModel>
	Particle(TDict const & dict, TModel const & model);

	// Destructor
	~Particle();

	template<typename TDict, typename TModel> void load(TDict const & dict, TModel const & model);

	template<typename ...Args>
	static std::shared_ptr<ParticleBase> create(Args && ... args)
	{
		return LoadParticle<this_type>(std::forward<Args>(args)...);
	}

	template<typename ...Args>
	static std::pair<std::string, std::function<std::shared_ptr<ParticleBase>(Args const &...)>> CreateFactoryFun()
	{
		std::function<std::shared_ptr<ParticleBase>(Args const &...)> call_back = []( Args const& ...args)
		{
			return this_type::create(args...);
		};
		return std::move(std::make_pair(get_type_as_string_static(), call_back));
	}
	//***************************************************************************************************
	// interface begin

	std::string save(std::string const & path) const;

	std::ostream& print(std::ostream & os) const
	{
		os << ",";
		properties.print(os);
		return os;
	}

	Real get_mass() const
	{
		return engine_type::get_mass();
	}

	Real get_charge() const
	{
		return engine_type::get_charge();
	}

	bool is_implicit() const
	{
		return engine_type::is_implicit;
	}

	std::string get_type_as_string() const
	{
		return get_type_as_string_static();
	}

	static std::string get_type_as_string_static()
	{
		return engine_type::get_type_as_string();
	}

	void const * get_rho() const
	{
		return reinterpret_cast<void const*>(&rho);
	}

	void const * get_J() const
	{
		return reinterpret_cast<void const*>(&J);
	}
	virtual void next_timestep();

	virtual void update_fields();

	// interface end
	//***************************************************************************************************

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

	typename mesh_type:: template field<VERTEX, scalar_type> rho;

	typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type> > J;

	std::list<std::function<void()> > constraint_;
};
template<typename TM, typename Engine>
Particle<TM, Engine>::Particle(mesh_type const & pmesh)
		: storage_type(pmesh), mesh(pmesh), rho(mesh), J(mesh)
{
}
template<typename TM, typename Engine>
template<typename TDict, typename TModel>
Particle<TM, Engine>::Particle(TDict const & dict, TModel const & model)
		: engine_type(dict), storage_type(model), mesh(model), rho(model), J(model)
{
	load(dict, model);
}
template<typename TM, typename Engine>
template<typename TDict, typename TModel>
void Particle<TM, Engine>::load(TDict const & dict, TModel const & model)
{
	engine_type::load(dict);

	storage_type::load(dict["url"].template as<std::string>());

	set_property("DumpParticle", dict["DumpParticle"].template as<bool>(false));

	set_property("DivergeJ", dict["DivergeJ"].template as<bool>(false));

	set_property("ScatterN", dict["ScatterN"].template as<bool>(false));

	storage_type::disable_sorting_ = dict["DisableSorting"].template as<bool>(false);

	J.clear();

	rho.clear();
}

template<typename TM, typename Engine>
Particle<TM, Engine>::~Particle()
{
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
void Particle<TM, Engine>::next_timestep()
{

	LOGGER << "Push particles to half step [ " << engine_type::get_type_as_string() << " ]";

	storage_type::Sort();

	Real dt = mesh.get_dt();

//	for (auto & cell : *this)
//	{
//		//TODO add rw cache
//		for (auto & p : cell.second)
//		{
//			this->engine_type::next_timestep(&p, dt, E0, B0, E1, B1);
//		}
//	}

	storage_type::set_changed();
}

template<typename TM, typename Engine>
void Particle<TM, Engine>::update_fields()
{

	LOGGER << "Scatter particles to fields [ " << engine_type::get_type_as_string() << " ]";

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

	if (properties["DivergeJ"].template as<bool>(false))
	{
		LOG_CMD(rho -= Diverge(MapTo<EDGE>(J)) * dt);
	}
	else if (properties["ScatterN"].template as<bool>(false))
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

#endif /* PARTICLE_H_ */
