/*
 * probe_particle.h
 *
 *  Created on: 2014年11月18日
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_PROBE_PARTICLE_H_
#define CORE_PARTICLE_PROBE_PARTICLE_H_

#include "tracable_particle.h"

namespace simpla
{

template<typename ...> struct _Particle;
template<typename TEngine, typename TDomain> using ProbeParticle
=_Particle<TEngine,TDomain,IsTracable>;

}  // namespace simpla

#endif /* CORE_PARTICLE_PROBE_PARTICLE_H_ */
