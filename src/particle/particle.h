/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_
#include "include/simpla_defs.h"
#include "datastruct/ndarray.h"
#include "datastruct/pool.h"
namespace simpla
{

template<typename TG, typename TS>
class Particle: public NdArray
{
public:

	typedef TG Grid;

	typedef TS ParticleType;

	typedef typename Grid::Index Index;

	typedef typename Grid::Coordinates Coordinates;

	typedef Particle<Grid, ParticleType> ThisType;

	const Grid & grid;

	typedef Pool<ParticleType> ParticlePool;

	typename Grid::ZeroForm n;

	Particle(const Grid &pgrid) :
			grid(pgrid)
	{

	}

	~Particle()
	{
	}

};

}  // namespace simpla

#endif /* PARTICLE_H_ */
