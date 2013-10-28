/*
 * particle_generator.h
 *
 *  Created on: 2013年10月24日
 *      Author: salmon
 */

#ifndef PARTICLE_GENERATOR_H_
#define PARTICLE_GENERATOR_H_

#include <particle/particle.h>

namespace simpla
{

template<typename T, typename XDIST, typename VDIST>
class ParticleGeneratorTraits
{
public:

	ParticleGeneratorTraits()
	{
	}

	template<typename Generator, typename T>
	inline void operator()(size_t num, Generator & g, T & res) const
	{
		res.x = x_dist(g)
		res.v = v_dist(g);
	}
private:
};

}  // namespace simpla

#endif /* PARTICLE_GENERATOR_H_ */
