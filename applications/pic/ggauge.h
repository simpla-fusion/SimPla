/*
 * ggauge.h
 *
 *  Created on: 2013年10月23日
 *      Author: salmon
 */

#ifndef GGAUGE_H_
#define GGAUGE_H_

#include "include/simpla_defs.h"
#include "numeric/multi_normal_distribution.h"
#include "numeric/normal_distribution_icdf.h"
#include "numeric/sobol_engine.h"
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
template<int NMATE, typename TG>
void init_load(Particle<GGauge_s<NMATE>, TG>& particle, int pic)
{


}

}  // namespace simpla

#endif /* GGAUGE_H_ */
