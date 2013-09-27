/*
 * Damping.h
 *
 *  Created on: 2012-10-30
 *      Author: salmon
 */

#ifndef DAMPING_H_
#define DAMPING_H_

namespace simpla
{

namespace field_fun
{

template<typename TV>
struct Damping
{

	Damping(PTree const pt)
	{
	}
	~Damping()
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

#endif /* DAMPING_H_ */
