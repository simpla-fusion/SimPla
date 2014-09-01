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
 */
template<typename TM, typename Engine> class KineticParticle;
template<typename TM, typename Engine> class FluidParticle;

template<typename TM, typename Engine, template<typename, typename > class ParticleConcept = KineticParticle>
class Particle: public ParticleConcept<TM, Engine>
{

public:
	static constexpr unsigned int IForm = VERTEX;

	typedef TM mesh_type;
	typedef Particle<TM, Engine> this_type;

	mesh_type const & mesh;

	template<typename ...Others>
	Particle(mesh_type const & pmesh, Others && ...); 	// Constructor

	~Particle();	// Destructor

	void load();

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others);

	std::string save(std::string const & path) const;

	std::ostream& print(std::ostream & os) const;

	template<typename ...Args> void next_timestep(Args && ...);

	template<typename TJ> void ScatterJ(TJ * J) const;

	template<typename TJ> void ScatterRho(TJ * rho) const;

};

}
// namespace simpla

#endif /* PARTICLE_H_ */
