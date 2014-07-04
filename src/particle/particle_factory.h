/*
 * particle_factory.h
 *
 *  Created on: 2014年6月13日
 *      Author: salmon
 */

#ifndef PARTICLE_FACTORY_H_
#define PARTICLE_FACTORY_H_

#include <string>

#include "../utilities/factory.h"
#include "particle_base.h"

namespace simpla
{

template<typename TM, typename ...Args> using ParticleFactory= Factory<std::string, ParticleBase<TM>, TM const&, Args && ...>;

}  // namespace simpla

#endif /* PARTICLE_FACTORY_H_ */
