/*
 * probe_particle.h
 *
 *  Created on: 2014年7月12日
 *      Author: salmon
 */

#ifndef PROBE_PARTICLE_H_
#define PROBE_PARTICLE_H_

#include <iostream>
#include <string>
#include <vector>

#include "../io/data_stream.h"
#include "../utilities/log.h"
#include "../utilities/primitives.h"
#include "../utilities/properties.h"
#include "../utilities/container_save_cache.h"

namespace simpla
{
class PolicyProbeParticle;
template<typename ...> class Particle;
template<typename TM, typename Engine> using ProbeParticle=Particle<TM, Engine, PolicyProbeParticle>;

template<typename TM, typename Engine>
struct Particle<TM, Engine, PolicyProbeParticle> : public Engine,
		public std::vector<typename Engine::Point_s>
{
	typedef TM manifold_type;
	typedef Engine engine_type;

	typedef Particle<manifold_type, engine_type, PolicyProbeParticle> this_type;

	typedef std::vector<typename Engine::Point_s> storage_type;

	typedef typename engine_type::Point_s particle_type;

	typedef typename engine_type::scalar_type scalar_type;

	typedef particle_type value_type;

	typedef typename manifold_type::iterator iterator;

	typedef typename manifold_type::coordinates_type coordinates_type;

	manifold_type const & mesh;

	//***************************************************************************************************
	// Constructor
	template<typename ...Others>
	Particle(manifold_type const & pmesh, Others && ...);	// Constructor

	// Destructor
	~Particle();

	static std::string get_type_as_string()
	{
		return "Probe" + engine_type::get_type_as_string();
	}

	void load();

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others);

	template<typename ...Args>
	std::string save(std::string const & path, Args &&...) const;

	std::ostream& print(std::ostream & os) const
	{
		engine_type::print(os);
		return os;
	}

	template<typename ...Args> void next_timestep(Real dt, Args && ...);

};

template<typename TM, typename Engine>
template<typename ... Others>
Particle<TM, Engine, PolicyProbeParticle>::Particle(manifold_type const & pmesh,
		Others && ...others) :
		mesh(pmesh)
{
	engine_type::load(std::forward<Others>(others)...);
}

template<typename TM, typename Engine>
Particle<TM, Engine, PolicyProbeParticle>::~Particle()
{
}

template<typename TM, typename Engine>
template<typename ...Args>
std::string Particle<TM, Engine, PolicyProbeParticle>::save(
		std::string const & path, Args && ...args) const
{
	return save(path, *this, std::forward<Args>(args)...);
}

template<typename TM, typename Engine>
template<typename ... Args>
void Particle<TM, Engine, PolicyProbeParticle>::next_timestep(Real dt,
		Args && ... args)
{

	LOGGER << "Push probe particles   [ " << get_type_as_string() << "]"
			<< std::endl;

	for (auto & p : *this)
	{
		this->engine_type::next_timestep(&p, dt, std::forward<Args>(args)...);
	}

	LOGGER << DONE;
}

}  // namespace simpla

#endif /* PROBE_PARTICLE_H_ */
