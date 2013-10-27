/*
 * particle_generator.h
 *
 *  Created on: 2013年10月24日
 *      Author: salmon
 */

#ifndef PARTICLE_GENERATOR_H_
#define PARTICLE_GENERATOR_H_

#include "particle.h"
namespace simpla
{

template<typename T, typename XDIST, typename YDIST, typename Generator>
void GenerateParticle(size_t num, XDIST & xdist, YDIST & ydist, Generator & g,
		PIC<T> * pic)
{
	for (size_t s = 0; s < num; ++s)
	{
		pic->emplace_back(x_dist(g), v_dist(g));
	}
}

}  // namespace simpla

#endif /* PARTICLE_GENERATOR_H_ */
