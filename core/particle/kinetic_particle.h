/**
 * \file kinetic_particle.h
 *
 * \date    2014年9月1日  下午2:25:26 
 * \author salmon
 */

#ifndef CORE_PARTICLE_KINETIC_PARTICLE_H_
#define CORE_PARTICLE_KINETIC_PARTICLE_H_

#include "untracable_particle.h"

namespace simpla
{
template<typename ...> class Particle;
template<typename TDomain, typename Engine> using KineticParticle=
Particle<TDomain, Engine, IsUntracable>;

}  // namespace simpla

#endif /* CORE_PARTICLE_KINETIC_PARTICLE_H_ */
