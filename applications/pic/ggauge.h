/*
 * ggauge.h
 *
 *  Created on: 2013年10月23日
 *      Author: salmon
 */

#ifndef GGAUGE_H_
#define GGAUGE_H_

#include <fetl/primitives.h>
namespace simpla
{

template<int NMATE>
struct GGauge_s
{
	Vec3 x;
	Vec3 v;
	Real f;
	Real w[NMATE];

};



template<typename T, typename XDIST, typename VDIST, typename RNDGenerator>
class ParticleGenerator;
template<int NMATE, typename XDIST, typename VDIST, typename RNDGenerator>
class ParticleGenerator<GGauge_s<NMATE>, XDIST, VDIST, RNDGenerator>
{
public:

	typedef typename GGauge_s<NMATE> value_type;
	typedef XDIST x_dist_type;
	typedef VDIST v_dist_type;
	typedef RNDGenerator generator_type;

	ParticleGenerator(x_dist_type && x_dist, v_dist_type && v_dist,
			RNDGenerator & g) :
			x_dist_(x_dist), v_dist_(v_dist), generator_(g)
	{
	}
	~ParticleGenerator()
	{
	}

	generator_type getGenerator() const
	{
		return generator_;
	}

	value_type operator()()
	{
		value_type res =
		{ x_dist_(generator_), v_dist_(generator_), 1.0 };
		std::fill(res->w[0], res->w + NMATE, 0);
		return std::move(res);
	}
private:
	x_dist_type x_dist_;
	v_dist_type v_dist_;
	generator_type generator_;
};

} // namespace simpla

#endif /* GGAUGE_H_ */
