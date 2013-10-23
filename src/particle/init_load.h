/*
 * init_load.h
 *
 *  Created on: 2013年10月18日
 *      Author: salmon
 */

#ifndef INIT_LOAD_H_
#define INIT_LOAD_H_

#include "numeric/sobol_engine.h"
#include "numeric/multi_normal_distribution.h"
#include "numeric/normal_distribution_icdf.h"

namespace simpla
{

template<typename Generator, typename XDIST, typename VDIST>
class InitLoad
{
	Vec3 xmin_, xmax_;
	Generator g_;
	XDIST x_dist_;
	VDIST v_dist_;
public:
	InitLoad(Generator g, XDIST x_dist, VDIST v_dist) :
			g_(g), x_dist_(x_dist), v_dist_(v_dist)
	{
	}
	template<typename IT>
	void operator()(IT & it)
	{
		it->x = x_dist_(g_);
		it->v = v_dist_(g_);
	}
};

void test()
{
	sobol_engine<6> g;
	multi_normal_distribution<3, double, normal_distribution_icdf<double> > v_dist_;

}
}  // namespace simpla

#endif /* INIT_LOAD_H_ */
