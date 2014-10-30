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
//#include "../utilities/container_save_cache.h"

namespace simpla
{
template<typename ...> class Particle;
template<typename TM, typename Engine> using ProbeParticle=Particle<Engine, TM>;

}  // namespace simpla

#endif /* PROBE_PARTICLE_H_ */
