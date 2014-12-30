/*
 * particle_factory.h
 *
 *  created on: 2014-6-13
 *      Author: salmon
 */

#ifndef PARTICLE_FACTORY_H_
#define PARTICLE_FACTORY_H_

#include <string>

#include "../core/design_pattern/factory.h"
#include "particle_base.h"

namespace simpla
{

template<typename TM, typename ...Args> using ParticleFactory=
Factory<std::string, ParticleBase , TM const&, Args && ...>;

}  // namespace simpla

#endif /* PARTICLE_FACTORY_H_ */
