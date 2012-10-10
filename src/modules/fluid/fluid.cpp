/*
 * fluid.cpp
 *
 *  Created on: 2011-12-10
 *      Author: salmon
 */
#include "defs.h"
#include "fluid.h"
#include "ohm_law.h"
namespace Fluid
{
void registerFunction(Context::Holder ctx)
{
	std::list<std::string> splist_;

	for (Context::SpeciesMap::iterator it = ctx->species.begin();
			it != ctx->species.end(); ++it)
	{
		if (boost::any_cast<std::string>(it->second["engine"]) == "ColdFluid")
		{
			splist_.push_back(it->first);
		}
	}

	if (!splist_.empty())
	{

		OhmLaw::Holder holder = OhmLaw::create(ctx);

		ctx->registerFunction(TR1::bind(&OhmLaw::pre_process, holder, splist_),
				"Pre-Process for Implict dJ/dt", -1);

		ctx->registerFunction(TR1::bind(&OhmLaw::process, holder),
				"Fluid push - Implicit dJ/dt = rho E + J x B", 0);

		ctx->registerFunction(TR1::bind(&OhmLaw::post_process, holder),

		"Post-Process for Implict dJ/dt", 1);
	}

}
}
