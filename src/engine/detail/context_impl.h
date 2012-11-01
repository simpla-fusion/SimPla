/*
 * context_impl.h
 *
 *  Created on: 2012-10-17
 *      Author: salmon
 */

#ifndef CONTEXT_IMPL_H_
#define CONTEXT_IMPL_H_

#include "fetl/fetl.h"
#include "modules/modules.h"
#include <boost/foreach.hpp>
namespace simpla
{

template<typename TG>
Context<TG>::Context()
{
	objFactory_["ZeroForm"] = TR1::bind(
			&Context<TG>::template CreateObject<ZeroForm>, this);

	objFactory_["OneForm"] = TR1::bind(
			&Context<TG>::template CreateObject<OneForm>, this);

	objFactory_["TwoForm"] = TR1::bind(
			&Context<TG>::template CreateObject<TwoForm>, this);

	objFactory_["ThreeForm"] = TR1::bind(
			&Context<TG>::template CreateObject<ThreeForm>, this);

	objFactory_["VecZeroForm"] = TR1::bind(
			&Context<TG>::template CreateObject<VecZeroForm>, this);

	objFactory_["VecOneForm"] = TR1::bind(
			&Context<TG>::template CreateObject<VecOneForm>, this);

	objFactory_["VecTwoForm"] = TR1::bind(
			&Context<TG>::template CreateObject<VecTwoForm>, this);

	objFactory_["VecThreeForm"] = TR1::bind(
			&Context<TG>::template CreateObject<VecThreeForm>, this);

	objFactory_["CZeroForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CZeroForm>, this);

	objFactory_["COneForm"] = TR1::bind(
			&Context<TG>::template CreateObject<COneForm>, this);

	objFactory_["CTwoForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CTwoForm>, this);

	objFactory_["CThreeForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CThreeForm>, this);

	objFactory_["CVecZeroForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CVecZeroForm>, this);

	objFactory_["CVecOneForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CVecOneForm>, this);

	objFactory_["CTwoForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CTwoForm>, this);

	objFactory_["CVecThreeForm"] = TR1::bind(
			&Context<TG>::template CreateObject<CVecThreeForm>, this);

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

	if (name != "")
	{
		std::map<std::string, TR1::shared_ptr<Object> >::iterator it =
				objects.find(name);
		if (it != objects.end())
		{
			if (it->second->CheckType(typeid(TOBJ)))
			{
				return TR1::dynamic_pointer_cast<TOBJ>(it->second);
			}
			else
			{
				ERROR << "The type of object " << name << " is not "
						<< typeid(TOBJ).name();
			}
		}
	}
	TR1::shared_ptr<TOBJ> res = CreateObject<TOBJ>();

	if (name != "")
	{
		objects[name] = TR1::dynamic_pointer_cast<ArrayObject>(res);
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
