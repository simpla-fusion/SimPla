/*
 * particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include "probe_particle.h"
#include "kinetic_particle.h"
#include "particle_engine.h"
namespace simpla
{

/** \defgroup  Particle Particle
 *
 *  \brief Particle  particle concept
 */

template<typename ...>struct Particle;

template<typename ...T>
std::ostream& operator<<(std::ostream & os, Particle<T...> const &p)
{
	p.print(os);
	return os;
}

}
// namespace simpla

#endif /* PARTICLE_H_ */
