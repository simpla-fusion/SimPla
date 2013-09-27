/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_
#include "include/simpla_defs.h"
#include "fetl/fetl.h"
#include "engine/object.h"
namespace simpla
{
template<typename TG>
struct Particle:public CompoundObject
{

	DEFINE_FIELDS(TG)
	ZeroForm n;
	VecZeroForm J;

};

}  // namespace simpla

#endif /* PARTICLE_H_ */
