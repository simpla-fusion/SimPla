/*
 * load_particle.h
 *
 *  Created on: 2013年12月21日
 *      Author: salmon
 */

#ifndef LOAD_PARTICLE_H_
#define LOAD_PARTICLE_H_

#include <random>

#include "../fetl/fetl.h"
#include "../numeric/multi_normal_distribution.h"
#include "../numeric/rectangle_distribution.h"

namespace simpla
{
template<typename > class Particle;

template<typename TEngine>
bool LoadParticle(size_t pic, Particle<TEngine> *p)
{

	typedef TEngine engine_type;

	typedef typename engine_type::mesh_type mesh_type;

	typedef Particle<engine_type> this_type;

	typedef typename engine_type::Point_s particle_type;

	typedef particle_type value_type;

	typedef typename mesh_type::scalar scalar;

	mesh_type const &mesh = p->mesh;

	std::mt19937 rnd_gen(1);

	rectangle_distribution<mesh_type::NUM_OF_DIMS> x_dist;

	multi_normal_distribution<mesh_type::NUM_OF_DIMS> v_dist(1.0);

	mesh.ParallelTraversal(Particle<TEngine>::IForm,

	[&](typename mesh_type::index_type const & s)
	{

		typename mesh_type::coordinates_type xrange[mesh.GetCellShape(s)];

		mesh.GetCellShape(s,xrange);

		x_dist.Reset(xrange);

		nTuple<3,Real> x,v;

		for(int i=0;i<pic;++i)
		{
			x_dist(rnd_gen,x);
			v_dist(rnd_gen,v);
			p->Insert(s,x,v,1.0);
		}
	}

	);

	return true;
}

template<typename TEngine, typename TN>
bool LoadParticle(size_t pic, Particle<TEngine> *p, TN const & n)
{

	typedef TEngine engine_type;

	typedef typename engine_type::mesh_type mesh_type;

	typedef Particle<engine_type> this_type;

	typedef typename engine_type::Point_s particle_type;

	typedef particle_type value_type;

	typedef typename mesh_type::scalar scalar;

	mesh_type const &mesh = p->mesh;

	std::mt19937 rnd_gen(1);

	rectangle_distribution<mesh_type::NUM_OF_DIMS> x_dist;

	multi_normal_distribution<mesh_type::NUM_OF_DIMS> v_dist(1.0);

	mesh.ParallelTraversal(Particle<TEngine>::IForm,

	[&](typename mesh_type::index_type const & s)
	{

		typename mesh_type::coordinates_type xrange[mesh.GetCellShape(s)];

		mesh.GetCellShape(s,xrange);

		x_dist.Reset(xrange);

		nTuple<3,Real> x,v;

		for(int i=0;i<pic;++i)
		{
			x_dist(rnd_gen,x);
			v_dist(rnd_gen,v);
			p->Insert(s,x,v,n(x));
		}
	}

	);

	return true;
}
}  // namespace simpla

#endif /* LOAD_PARTICLE_H_ */
