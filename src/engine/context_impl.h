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
#include "modules/em/pml.h"
#include "modules/fluid/cold_fluid.h"
#include "modules/io/write_xdmf.h"
#include <boost/foreach.hpp>
namespace simpla
{

template<typename TG>
template<typename PT>
TR1::shared_ptr<Context<TG> > Context<TG>::Create(PT const & pt)
{
	return TR1::shared_ptr<Context<TG> >(new ThisType(pt));
}

template<typename TG>
inline std::string Context<TG>::Summary() const
{
	std::ostringstream os;

	os

	<< PHYS_CONSTANTS.Summary()

	<< SINGLELINE << std::endl

	<< std::setw(20) << "dt : " << dt << std::endl

	<< grid.Summary() << std::endl

	<< DOUBLELINE << std::endl

	;


//	<< "Objects List" << std::endl
//
//	<< SINGLELINE << std::endl
//
//
//	for (typename std::map<std::string, TR1::shared_ptr<Object> >::const_iterator it =
//			objects.begin(); it != objects.end(); ++it)
//	{
//		os << it->first << std::endl;
//	}

	return os.str();

}
template<typename TG>
template<typename PT>
void Context<TG>::LoadModules(PT const & pt)
{
	LOG << "Load Modules";

	BOOST_FOREACH(const typename PT::value_type &v, pt.get_child("Modules"))
	{
		boost::optional<std::string> type = v.second.template get_optional<
				std::string>("<xmlattr>.type");
		if (!type)
		{
			continue;
		}
		if (*type == "Maxwell")
		{
			modules.push_back(
					TR1::bind(&em::Maxwell<TG>::Eval,
							new em::template Maxwell<TG>(*this, v.second)));
		}
		else if (*type == "PML")
		{
			modules.push_back(
					TR1::bind(&em::PML<TG>::Eval,
							new em::template PML<TG>(*this, v.second)));
		}
		else if (*type == "ColdFluid")
		{
			modules.push_back(
					TR1::bind(&em::ColdFluid<TG>::Eval,
							new em::template ColdFluid<TG>(*this, v.second)));
		}

	}

	boost::optional<std::string> type = pt.template get_optional<std::string>(
			"OutPut.<xmlattr>.type");

	if (!!type)
	{
		if (*type == "XDMF")
		{
			modules.push_back(
					TR1::bind(&io::WriteXDMF<TG>::Eval,
							new io::template WriteXDMF<TG>(*this,
									pt.get_child("OutPut"))));

		}
	}
}

}  // namespace simpla

#endif /* CONTEXT_IMPL_H_ */
