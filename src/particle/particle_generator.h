/*
 * particle_generator.h
 *
 *  Created on: 2013年10月24日
 *      Author: salmon
 */

#ifndef PARTICLE_GENERATOR_H_
#define PARTICLE_GENERATOR_H_

namespace simpla
{
template<typename T, typename XDIST, typename VDIST>
class ParticleGenerator
{
public:

	typedef T value_type;
	typedef XDIST x_dist_type;
	typedef VDIST v_dist_type;

	ParticleGenerator(x_dist_type && x_dist, v_dist_type && v_dist) :
			x_dist_(x_dist), v_dist_(v_dist)
	{
	}
	~ParticleGenerator()
	{
	}
	template<typename Generator>
	T && operator()(Generator & g)
	{
		T(x_dist_(g), v_dist_(g));
	}
private:
	x_dist_type x_dist_;
	v_dist_type v_dist_;

};

}  // namespace simpla

#endif /* PARTICLE_GENERATOR_H_ */
