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
#include "../utilities/properties.h"
#include "../parallel/parallel.h"
#include "../model/model.h"

#include "particle_base.h"
#include "particle_pool.h"
#include "load_particle.h"
#include "save_particle.h"

#include "particle_boundary.h"

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
template<class Engine>
class Particle: public ParticleBase, public Engine, public ParticlePool<typename Engine::mesh_type,
        typename Engine::Point_s>
{

public:
	static constexpr unsigned int IForm = VERTEX;

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
	Properties properties;
	mesh_type const & mesh;
	//***************************************************************************************************
	// Constructor
	Particle(mesh_type const & pmesh);

	template<typename TDict> Particle(TDict const & dict, mesh_type const & pmesh);

	// Destructor
	~Particle();

	template<typename TDict> void load(TDict const & dict);

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

	void set_property_(std::string const & name, Any const&v)
	{
		properties[name] = v;
	}
	Any const & get_property_(std::string const &name) const
	{
		return properties[name].value();
	}

	template<typename T> void set_property(std::string const & name, T const&v)
	{
		set_property_(name, Any(v));
	}

	template<typename T> T get_property(std::string const & name) const
	{
		return get_property_(name).template as<T>();
	}

	bool same_mesh_type(std::type_info const & t_info) const
	{
		return t_info == typeid(mesh_type);
	}

	std::string save(std::string const & path) const;

	std::ostream& print(std::ostream & os) const
	{
		engine_type::print(os);
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

	void const * get_n() const
	{
		return reinterpret_cast<void const*>(&n);
	}

	void const * get_J() const
	{
		return reinterpret_cast<void const*>(&J);
	}

	void next_timestep_zero_(void const * E, void const*B)
	{
		next_timestep_zero(*reinterpret_cast<E_type const*>(E), *reinterpret_cast<B_type const*>(B));
	}

	void next_timestep_half_(void const * E, void const*B)
	{
		next_timestep_half(*reinterpret_cast<E_type const*>(E), *reinterpret_cast<B_type const*>(B));
	}

	void update_fields();

	//***************************************************************************************************
	// interface end

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

	typedef typename engine_type::E_type E_type;
	typedef typename engine_type::B_type B_type;

	void next_timestep_zero(E_type const &E, B_type const & B);
	void next_timestep_half(E_type const &E, B_type const & B);

	std::list<std::function<void()> > constraint_;
};
template<typename Engine>
Particle<Engine>::Particle(mesh_type const & pmesh) :
		engine_type(pmesh), storage_type(pmesh), mesh(pmesh), n(mesh), J(mesh)
{
}
template<typename Engine>
template<typename TDict>
Particle<Engine>::Particle(TDict const & dict, mesh_type const & pmesh) :
		Particle(pmesh)
{
	load(dict);
}
template<typename Engine>
template<typename TDict>
void Particle<Engine>::load(TDict const & dict)
{
	engine_type::load(dict);

	storage_type::load(dict["url"].template as<std::string>());

	set_property("DumpParticle", dict["DumpParticle"].template as<bool>(false));

	J.clear();

	n.clear();
}

template<typename Engine>
Particle<Engine>::~Particle()
{
}

template<typename ...T>
std::ostream &operator<<(std::ostream&os, Particle<T...> const & p)
{
	p.print(os);
	return os;
}

//*************************************************************************************************

template<typename Engine>
std::string Particle<Engine>::save(std::string const & path) const
{
	std::stringstream os;

	GLOBAL_DATA_STREAM.cd(path);

	os << "\n, n =" << simpla::save("n", n);

	os << "\n, J =" << simpla::save("J", J);

	if (get_property<bool>("DumpParticle"))
	{
		os << "\n, particles = " << storage_type::save("particles");
	}

	return os.str();
}

template<typename Engine>
void Particle<Engine>::next_timestep_zero(E_type const & E, B_type const & B)
{

	VERBOSE << "Push particles to zero step [ " << engine_type::get_type_as_string() << " ]";

	storage_type::Sort();

	Real dt = mesh.get_dt();

	J.clear();

	for (auto & cell : *this)
	{
		//TODO add rw cache
		for (auto & p : cell.second)
		{
			this->engine_type::next_timestep_zero(&p, dt, E, B);
		}
	}

	storage_type::EnableSort();
}
template<typename Engine>
void Particle<Engine>::next_timestep_half(E_type const & E, B_type const & B)
{

	VERBOSE << "Push particles to half step [ " << engine_type::get_type_as_string() << " ]";

	storage_type::Sort();

	Real dt = mesh.get_dt();

	J.clear();

	for (auto & cell : *this)
	{
		//TODO add rw cache
		for (auto & p : cell.second)
		{
			this->engine_type::next_timestep_half(&p, dt, E, B);
		}
	}

	storage_type::EnableSort();
}

template<typename Engine>
void Particle<Engine>::update_fields()
{
	VERBOSE << "Scatter particles to fields ";

//	VERBOSE << "Scatter particles to fields [ " << engine_type::get_type_as_string() << " ]";

	Real dt = mesh.get_dt();

	storage_type::Sort();

	J.clear();

	for (auto & cell : *this)
	{
		//TODO add rw cache
		for (auto & p : cell.second)
		{
			this->engine_type::Scatter(p, &J);
		}
	}

	UpdateGhosts(&J);

	if (properties["Update Density"].template as<bool>(true))
	{
		VERBOSE_CMD(n -= Diverge(MapTo<EDGE>(J)) * dt);
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
