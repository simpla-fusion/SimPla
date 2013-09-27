/*
 * smooth.h
 *
 *  Created on: 2012-10-30
 *      Author: salmon
 */

#ifndef SMOOTH_H_
#define SMOOTH_H_
#include "include/simpla_defs.h"
#include "utilities/properties.h"

namespace simpla
{

namespace field_fun
{

template<typename TV>
struct Smooth
{

	Smooth(PTree const pt)
	{
	}
	~Smooth()
	{
	}
	template<typename TE>
	TV operator()(nTuple<THREE, TE> x, Real t)
	{
		return TV();
	}

};

}  // namespace field_fun

}  // namespace simpla

#endif /* SMOOTH_H_ */
