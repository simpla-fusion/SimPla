/*
 * ramp_wave.h
 *
 *  Created on: 2012-10-30
 *      Author: salmon
 */

#ifndef RAMP_WAVE_H_
#define RAMP_WAVE_H_

#include "include/simpla_defs.h"
#include "utilities/properties.h"
#include "fetl/grid.h"
#include "fetl/fetl.h"
#include "engine/context.h"
namespace simpla
{

namespace field_fun
{

template<typename TV>
struct RampWave
{
	TV A;
	Real alpha;
	Real freq;
	RVec3 k;
	RampWave(ptree const pt) :
			alpha(pt.get("alpha", 1.0e128)),

			freq(pt.get("freq", 0.0d)),

			k(pt.get<RVec3>("k", pt_trans<RVec3, std::string>())),

			A(pt.get<TV>("amp", pt_trans<TV, std::string>()))

	{
	}
	~RampWave()
	{
	}
	template<typename TE>
	TV operator()(nTuple<THREE, TE> x, Real t)
	{
		return A * (1.0 - std::exp(-t * alpha)) * std::sin(freq * t + Dot(k, x));
	}

};

}  // namespace field_fun

}  // namespace simpla
#endif /* RAMP_WAVE_H_ */
