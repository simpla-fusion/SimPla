/*
 * pic.cpp
 *
 *  Created on: 2011-7-5
 *      Author: salmon
 */
#include "defs.h"
#include "pic.h"
#include "gyro_gauge.h"
#include "delta_f.h"
#include <vector>
#include <list>
namespace PIC
{
void registerFunction(Context::Holder ctx)
{

	for (Context::SpeciesMap::iterator it = ctx->species.begin();
			it != ctx->species.end(); ++it)
	{

		if (boost::any_cast<std::string>(it->second["engine"]) == "GyroGauge")
		{

			ctx->registerFunction(
					TR1::bind(&PIC::GyroGauge::pre_process, ctx, it->first),
					"Init Load  -- GyroGauge [" + it->first + "]",
					Context::PRE_PROCESS);

			ctx->registerFunction(
					TR1::bind(&PIC::GyroGauge::process, ctx, it->first),
					"PIC push J -- GyroGauge [" + it->first + "]",
					Context::PROCESS);

		}
		else if (boost::any_cast<std::string>(it->second["engine"]) == "DeltaF")
		{

			ctx->registerFunction(TR1::bind(&PIC::DeltaF::init, ctx, it->first),
					"Init. Load-- Delta-f [" + it->first + "]",
					Context::PRE_PROCESS);

			ctx->registerFunction(TR1::bind(&PIC::DeltaF::push, ctx, it->first),
					"PIC push J -- Delta-f [" + it->first + "]",
					Context::PROCESS);
		}

	}
//		else if (sp->engineName == "HybridDeltaF")
//		{
//			auto holder = PIC::DeltaF::create(ctx, sp);
//			holder->initLoad(pic);
//			ctx->registerFunction(50, holder, &PIC::DeltaF::pushDivStress,
//					"PIC push DivStress -- Delta-f [" + sp->name + "]");
//			ctx->fluidList.push_back(sp);
//		}
}

} // namespace PIC
