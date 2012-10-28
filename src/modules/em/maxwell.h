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
#include "engine/basecontext.h"
#include "engine/modules.h"
#include "utilities/properties.h"
namespace simpla
{
namespace em
{

template<typename TG>
class Maxwell: public Module
{
public:

	typedef Maxwell<TG> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	DEFINE_FIELDS(typename TG::ValueType, TG)

	BaseContext & ctx;

	Maxwell(BaseContext & d, ptree const &) :
			ctx(d),

			dt(ctx.dt),

			mu0(ctx.PHYS_CONSTANTS["permeability_of_free_space"]),

			epsilon0(ctx.PHYS_CONSTANTS["permittivity_of_free_space"]),

			speed_of_light(ctx.PHYS_CONSTANTS["speed_of_light"]),

			B1(ctx.template GetObject<TwoForm>("B1")),

			E1(ctx.template GetObject<OneForm>("E1")),

			J1(ctx.template GetObject<OneForm>("J1"))
	{
		LOG << "Create module Maxwell";
	}

	virtual ~Maxwell()
	{
	}

	virtual void Eval()
	{
		LOG << "Run module Maxwell";

		*B1 -= Curl(*E1) * dt;

		*E1 += (Curl(*B1 / mu0) - *J1) / epsilon0 * dt;
	}

private:
	const Real dt;
	const Real mu0;
	const Real epsilon0;
	const Real speed_of_light;

//input
	TR1::shared_ptr<const OneForm> J1;
//output
	TR1::shared_ptr<OneForm> E1;
	TR1::shared_ptr<TwoForm> B1;

};

} // namespace em_field
} // namespace simpla
#endif  // SRC_EMFIELD_MAXWELL_H_
