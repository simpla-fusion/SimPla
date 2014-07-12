/*
 * probe_particle.h
 *
 *  Created on: 2014年7月12日
 *      Author: salmon
 */

#ifndef PROBE_PARTICLE_H_
#define PROBE_PARTICLE_H_
#include <exception>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../fetl/fetl.h"
#include "../utilities/log.h"
#include "../utilities/sp_type_traits.h"

#include "../parallel/parallel.h"
#include "../model/model.h"

#include "particle_base.h"
#include "load_particle.h"
#include "save_particle.h"

namespace simpla
{

template<typename Engine>
class ProbeParticle: public Engine, public ParticleBase, public std::vector<typename Engine::Point_s>
{
public:

	typedef Engine engine_type;

	typedef ProbeParticle<engine_type> this_type;

	typedef ContainerSaveCache<typename Engine::Point_s> storage_type;

	typedef typename engine_type::mesh_type mesh_type;

	typedef typename engine_type::Point_s particle_type;

	typedef typename engine_type::scalar_type scalar_type;

	typedef particle_type value_type;

	typedef typename mesh_type::iterator iterator;

	typedef typename mesh_type::coordinates_type coordinates_type;

public:
	//***************************************************************************************************
	// Constructor
	ProbeParticle();

	template<typename TDict, typename ...Others> ProbeParticle(TDict const & dict, Others &&...othes);

	// Destructor
	~ProbeParticle();

	template<typename TDict> void load(TDict const & dict);

	template<typename ...Args>
	static std::shared_ptr<ParticleBase> create(Args && ... args)
	{
		return std::dynamic_pointer_cast<ParticleBase>(
		        std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...)));
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

	bool same_mesh_type(std::type_info const & t_info) const
	{
		return t_info == typeid(mesh_type);
	}

	std::string save(std::string const & path) const;

	std::ostream& print(std::ostream & os) const
	{
		engine_type::print(os);
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

	typedef typename mesh_type::template field<EDGE, scalar_type> E_type;
	typedef typename mesh_type::template field<FACE, scalar_type> B_type;

	void next_timestep_zero_(void const * E, void const*B)
	{
		next_timestep_zero(*reinterpret_cast<E_type const *>(E), *reinterpret_cast<B_type const*>(B));
	}

	void next_timestep_half_(void const * E, void const*B)
	{
		next_timestep_half(*reinterpret_cast<E_type const*>(E), *reinterpret_cast<B_type const*>(B));
	}

	template<typename TE, typename TB>
	void next_timestep_zero(TE const &E, TB const & B);

	template<typename TE, typename TB>
	void next_timestep_half(TE const &E, TB const & B);

};
template<typename Engine>
Particle<Engine>::Particle(mesh_type const & pmesh) :
		engine_type(pmesh), mesh(pmesh)
{
}
template<typename Engine>
template<typename TDict, typename ...Others>
ProbeParticle<Engine>::ProbeParticle(TDict const & dict, Others &&...others) :
		ProbeParticle(pmesh)
{
	load(dict);
}
template<typename Engine>
template<typename TDict>
void ProbeParticle<Engine>::load(TDict const & dict)
{
	engine_type::load(dict);
}

template<typename Engine>
ProbeParticle<Engine>::~ProbeParticle()
{
}

template<typename Engine>
std::string ProbeParticle<Engine>::save(std::string const & path) const
{
	std::stringstream os;

	GLOBAL_DATA_STREAM.OpenGroup(path);

//	if (is_verbose)
//	{
//		GLOBAL_DATA_STREAM.DisableCompactStorable();
//		os
//
//		<< engine_type::save(path)
//
//		<< "\n, ProbeParticles = " << storage_type::save("particles")
//
//		;
//
//		os << "\n, n =" << simpla::save( "n",n);
//
//		os << "\n, J =" << simpla::save( "J",J);
//
//		GLOBAL_DATA_STREAM.EnableCompactStorable();
//	}
//	else

	return "";
}

template<typename Engine>
template<typename TE, typename TB>
void Particle<Engine>::next_timestep_zero(TE const & E, TB const & B)
{

	LOGGER << "Push probe particles to zero step [ " << engine_type::get_type_as_string();

	Real dt = mesh.get_dt();

	for (auto & p : *this)
	{
		this->engine_type::next_timestep_zero(&p, dt, E, B);
	}

	LOGGER << DONE;

}

template<typename Engine>
template<typename TE, typename TB>
void Particle<Engine>::next_timestep_half(TE const & E, TB const & B)
{

	LOGGER << "Push probe particles to half step[ " << engine_type::get_type_as_string();

	Real dt = mesh.get_dt();

	for (auto & p : *this)
	{
		this->engine_type::next_timestep_half(&p, dt, E, B);
	}

	LOGGER << DONE;
}

}  // namespace simpla

#endif /* PROBE_PARTICLE_H_ */
