/*
 * context_impl.h
 *
 *  Created on: 2012-10-17
 *      Author: salmon
 */

#ifndef CONTEXT_IMPL_H_
#define CONTEXT_IMPL_H_
#include "context.h"
#include "modules/em/maxwell.h"
#include "modules/fluid/cold_fluid.h"

namespace simpla
{

template<typename TG>
void Context<TG>::LoadModules(ptree const & pt)
{
	std::pair<ptree::const_assoc_iterator, ptree::const_assoc_iterator> it_range =
			pt.equal_range("module");
	for (typename ptree::const_assoc_iterator it = it_range.first;
			it != it_range.second; ++it)
	{
		std::string type = it->second.get < std::string > ("<xmlattr>.type");

		if (type == "Maxwell")
		{
			modules.push_back(
					TR1::bind(&em::Maxwell<Real, Grid>::Eval,
							new em::Maxwell<Real, Grid>(*this, it->second)));
		}
		else if (type == "ColdFluid")
		{
			modules.push_back(
					TR1::bind(&em::ColdFluid<Real, Grid>::Eval,
							new em::ColdFluid<Real, Grid>(*this, it->second)));
		}
	}

	//template<typename TG>
	//void RegisterModules(ptree const & pt)
	//{

	//
	//}

}

}  // namespace simpla

#endif /* CONTEXT_IMPL_H_ */
