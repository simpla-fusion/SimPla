/*
 * context_impl.h
 *
 *  Created on: 2012-10-17
 *      Author: salmon
 */

#ifndef CONTEXT_IMPL_H_
#define CONTEXT_IMPL_H_

#include "fetl/fetl.h"
#include "particle/particle.h"
#include "modules/modules.h"
#include <boost/foreach.hpp>
namespace simpla
{

template<typename TG>
Context<TG>::Context()
{
	RegisterFields(this);
	RegisterParticles<TG>(this);
	RegisterModules(this);

}
template<typename TG>
Context<TG>::~Context()
{
}
template<typename TG>
void Context<TG>::Parse(ptree const & pt)
{
	grid.Parse(pt.get_child("Grid"));
	BaseContext::Parse(pt);
}
template<typename TG>
template<typename TOBJ>
TR1::shared_ptr<TOBJ> Context<TG>::CreateObject()
{
	return TR1::shared_ptr<TOBJ>(new TOBJ(grid));
}

template<typename TG>
template<typename TOBJ>
TR1::shared_ptr<TOBJ> Context<TG>::GetObject(std::string const & name)
{
	if (name == "")
	{
		ERROR << "The name of  object   is empty!";
	}

	TR1::shared_ptr<TOBJ> res;

	if (boost::optional<TR1::shared_ptr<Object> > obj = objects->FindObject(
			name))
	{

		if ((*obj)->CheckType(typeid(TOBJ)))
		{
			res = TR1::dynamic_pointer_cast<TOBJ>(*obj);
		}
		else
		{
			ERROR << "The type of object " << name << " is not "
					<< typeid(TOBJ).name();
		}
	}
	else
	{

		res = CreateObject<TOBJ>();

		objects->operator[](name) = TR1::dynamic_pointer_cast<Object>(res);

	}
	return res;
}
//template<typename TG>
//TR1::shared_ptr<Object> Context<TG>::ObjectFactory(std::string const & type)
//{
//	TR1::shared_ptr<Object> res;
//	if (type == "ZeroForm")
//	{
//		res = CreateObject<ZeroForm>();
//	}
//	else if (type == "OneForm")
//	{
//		res = CreateObject<OneForm>();
//	}
//	else if (type == "TwoForm")
//	{
//		res = CreateObject<TwoForm>();
//	}
//	else if (type == "ThreeForm")
//	{
//		res = CreateObject<ThreeForm>();
//	}
//	else if (type == "VecZeroForm")
//	{
//		res = CreateObject<VecZeroForm>();
//	}
//	else if (type == "VecOneForm")
//	{
//		res = CreateObject<VecOneForm>();
//
//	}
//	else if (type == "TwoForm")
//	{
//		res = CreateObject<VecTwoForm>();
//	}
//	else if (type == "VecThreeForm")
//	{
//		res = CreateObject<VecThreeForm>();
//	}
//	else if (type == "CZeroForm")
//	{
//		res = CreateObject<CZeroForm>();
//	}
//	else if (type == "COneForm")
//	{
//		res = CreateObject<COneForm>();
//	}
//	else if (type == "CTwoForm")
//	{
//		res = CreateObject<CTwoForm>();
//	}
//	else if (type == "CThreeForm")
//	{
//		res = CreateObject<CThreeForm>();
//	}
//	else if (type == "CVecZeroForm")
//	{
//
//		res = CreateObject<CVecZeroForm>();
//	}
//	else if (type == "CVecOneForm")
//	{
//
//		res = CreateObject<CVecOneForm>();
//
//	}
//	else if (type == "CTwoForm")
//	{
//
//		res = CreateObject<CVecTwoForm>();
//
//	}
//	else if (type == "CVecThreeForm")
//	{
//
//		res = CreateObject<CVecThreeForm>();
//	}
//	else
//	{
//		res = BaseContext::ObjectFactory(type);
//	}
//	return res;
//}

//template<typename TG>
//TR1::function<void(void)> Context<TG>::ModuleFactory(std::string const & name,
//		ptree const & pt)
//{
//
//	TR1::function<void(void)> res;
//	if (name == "RFSrc")
//	{
//		res = TR1::bind(&em::RFSrc<TG>::Eval,
//				new em::template RFSrc<TG>(*this, pt));
//	}
//	else if (name == "Maxwell")
//	{
//		res = TR1::bind(&em::Maxwell<TG>::Eval,
//				new em::template Maxwell<TG>(*this, pt));
//	}
//	else if (name == "PML")
//	{
//		res = TR1::bind(&em::PML<TG>::Eval,
//				new em::template PML<TG>(*this, pt));
//	}
//	else if (name == "ColdFluid")
//	{
//		res = TR1::bind(&em::ColdFluid<TG>::Eval,
//				new em::template ColdFluid<TG>(*this, pt));
//	}
//	else if (name == "XDMF")
//	{
//		res = TR1::bind(&io::WriteXDMF<TG>::Eval,
//				new io::template WriteXDMF<TG>(*this, pt));
//	}
//	else
//	{
//		BaseContext::ModuleFactory(name, pt);
//	}
//
//}
template<typename TG>
inline std::string Context<TG>::Summary() const
{
	std::ostringstream os;

	os

	<< PHYS_CONSTANTS.Summary()

	<< SINGLELINE << std::endl

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

} // namespace simpla

#endif /* CONTEXT_IMPL_H_ */
