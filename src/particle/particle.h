/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_
#include "include/simpla_defs.h"
//#include "datastruct/ndarray.h"
//#include "datastruct/pool.h"
#include "fetl/fetl.h"
namespace simpla
{
template<typename TG>
struct Particle
{
	DEFINE_FIELDS(TG);

	Real m;
	Real Z;
	Real T;

	ZeroForm n;
	VecZeroForm J;



};

}  // namespace simpla

#endif /* PARTICLE_H_ */
