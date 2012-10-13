/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * Maxwell/Maxwell.h
 *
 *  Created on: 2010-11-16
 *      Author: salmon
 */

#ifndef SRC_EMFIELD_MAXWELL_H_
#define SRC_EMFIELD_MAXWELL_H_

#include "fetl/fetl.h"
#include "fetl/vector_calculus.h"
#include "engine/solver.h"
#include "engine/modules.h"

namespace simpla
{
namespace em
{

template<typename TV, typename TG>
class Maxwell: public Modules
{
public:

	typedef Maxwell ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	DEFINE_FIELDS(TV, TG)

	Maxwell(Domain & d, const ptree & properties) :
			Modules(d),

			dt(d.dt),

			mu0(d.PHYS_CONSTANTS.get<Real>("mu")),

			epsilon0(d.PHYS_CONSTANTS.get<Real>("epsilon")),

			speed_of_light(d.PHYS_CONSTANTS.get<Real>("speed_of_light")),

			B1(d.GetObject<TwoForm>("B1")),

			E1(d.GetObject<OneForm>("E1")),

			J1(d.GetObject<OneForm>("J1"))
	{
	}

	virtual ~Maxwell()
	{
	}

	virtual void Eval()
	{

		B1 -= Curl(E1) * dt;

		E1 += (Curl(B1 / mu0) - J1) / epsilon0 * dt;
	}

private:
	const Real dt;
	const Real mu0;
	const Real epsilon0;
	const Real speed_of_light;

//input
	OneForm const & J1;
//output
	OneForm & E1;
	TwoForm & B1;

};

} // namespace em_field
} // namespace simpla
#endif  // SRC_EMFIELD_MAXWELL_H_
