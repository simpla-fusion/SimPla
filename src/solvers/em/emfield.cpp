/*
 * emfield.cpp
 *
 *  Created on: 2011-12-10
 *      Author: salmon
 */
#include "defs.h"
#include "emfield.h"

#include "pml.h"
#include "maxwell.h"

namespace em_field
{

void registerFunction(Context::Holder ctx)
{

//		if (cfgPaser.check("PERMITTIVITY") && cfgPaser.check("PERMEABILITY"))
//		{
//			auto epsilon = Context::ZeroForm::create(ctx->grid);
//			auto mu = Context::ZeroForm::create(ctx->grid);
//
//			ctx->registerObject(epsilon, "epsilon");
//			ctx->registerObject(mu, "mu");
//
//			cfgPaser["PERMITTIVITY"].to(*epsilon);
//			cfgPaser["PERMEABILITY"].to(*mu);
//		}

	bool hasPML = false;

	for (int i = 0; i < 3; ++i)
	{
		hasPML |= (ctx->bc[i * 2] > 1 && ctx->grid->dims[i] > 1);
	}

	if (hasPML)
	{
		PML::Holder holder = PML::create(ctx);
		ctx->registerFunction(TR1::bind(&PML::pre_process, holder),
				"Prepare for PML", -1);
		ctx->registerFunction(TR1::bind(&PML::process, holder),
				"PML dB/dt=-Curl(E),dD/dt=Curl(H)-J", 0);
		ctx->registerFunction(TR1::bind(&PML::post_process, holder),
				"Post-process for PML", 1);
	}
	else
	{
		Maxwell::Holder holder = Maxwell::create(ctx);

		ctx->registerFunction(TR1::bind(&Maxwell::pre_process, holder),
				"Prepare Maxwell Eqs.", -1);
		ctx->registerFunction(TR1::bind(&Maxwell::process, holder),
				"dB/dt=-Curl(E),dD/dt=Curl(H)-J", 0);
		ctx->registerFunction(TR1::bind(&Maxwell::post_process, holder),
				"Maxwell Eq", 1);
	}

}

} /* namespace EMField */
