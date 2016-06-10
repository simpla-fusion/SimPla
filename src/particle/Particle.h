/**
 * @file Particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_PARTICLE_H_
#define CORE_PARTICLE_PARTICLE_H_

//#include "ParticleV000.h"
#include "ParticleV001.h"

namespace simpla { namespace particle
{

template<typename P, typename M> using DefaultParticle=Particle<P, M, V001>;

}}//namespace simpla { namespace particle

#endif /* CORE_PARTICLE_PARTICLE_H_ */
