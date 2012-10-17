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

template<typename TOBJ>
TR1::shared_ptr<TOBJ> BaseContext::GetObject(std::string const & name)
{

	if (name != "")
	{
		std::map<std::string, Object::Holder>::iterator it = objects.find(name);
		if (it != objects.end())
		{
			if (it->second->CheckType(typeid(TOBJ)))
			{
				return TR1::dynamic_pointer_cast<TOBJ>(it->second);
			}
			else
			{
				ERROR << "Object " << name << "can not been created as "
						<< typeid(TOBJ).name();
			}
		}
	}

	TR1::shared_ptr<TOBJ> res(
			new TOBJ(*static_cast<typename TOBJ::Grid const *>(getGridPtr())));
	if (name != "")
	{
		objects[name] = TR1::dynamic_pointer_cast<Object>(res);
	}

	return res;
}

template<typename TG>
TR1::shared_ptr<Context<TG> > Context<TG>::Create(ptree const & pt)
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

	<< "Objects List" << std::endl

	<< SINGLELINE << std::endl

	;
	for (typename std::map<std::string, TR1::shared_ptr<Object> >::const_iterator it =
			objects.begin(); it != objects.end(); ++it)
	{
		os << it->first << std::endl;
	}

	return os.str();

}

template<typename TG>
void Context<TG>::LoadModules(ptree const & pt)
{
	LOG << "Load Modules";

	BOOST_FOREACH(const ptree::value_type &v, pt.get_child("Modules"))
	{
		std::string type = v.second.get<std::string>("<xmlattr>.type");

		if (type == "Maxwell")
		{
			modules.push_back(
					TR1::bind(&em::Maxwell<Real, Grid>::Eval,
							new em::Maxwell<Real, Grid>(*this, v.second)));
		}
		else if (type == "PML")
		{
			modules.push_back(
					TR1::bind(&em::PML<Real, Grid>::Eval,
							new em::PML<Real, Grid>(*this, v.second)));
		}
		else if (type == "ColdFluid")
		{
			modules.push_back(
					TR1::bind(&em::ColdFluid<Real, Grid>::Eval,
							new em::ColdFluid<Real, Grid>(*this, v.second)));
		}
		else if (type == "WriteXDMF")
		{
			modules.push_back(
					TR1::bind(&io::WriteXDMF<Grid>::Eval,
							new io::WriteXDMF<Grid>(*this, v.second)));
		}
	}

}

}  // namespace simpla

#endif /* CONTEXT_IMPL_H_ */
