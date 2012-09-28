/*
 * Particle.h
 *
 *  Created on: 2011-12-14
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <vector>
#include "include/simpla_defs.h"
#include "engine/object.h"
#include "fetl/fetl.h"
#include "pic/particle_pool.h"
#include "pic/delta_f.h"
#include "pic/full_f.h"
namespace simpla {
namespace pic {
template<typename, typename > struct PICEngine;
template<typename TS, typename TG> struct ParticlePool;
} // namespace pic
} // namespace simpla

#endif /* PARTICLE_H_ */
