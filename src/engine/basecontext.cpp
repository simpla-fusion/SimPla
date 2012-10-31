/*
 * domain.cpp
 *
 *  Created on: 2012-10-13
 *      Author: salmon
 */
#include <iostream>
#include "include/simpla_defs.h"
#include "context.h"
#include "object.h"
#include "fetl/grid.h"
#include "modules/flow_control/flow_control.h"
namespace simpla
{

BaseContext::BaseContext() :
		dt(0.0), counter_(0), timer_(0), output_path("")
{
}
BaseContext::BaseContext(ptree const&pt) :
		dt(pt.get("Grid.Time.<xmlattr>.dt", 1.0d)),

		PHYS_CONSTANTS(pt.get_child("PhysConstants")),

		counter_(0), timer_(0), output_path("Untitled")
{
	moduleFactory_["Loop"] = TR1::bind(&flow_control::Loop::Create, this,
			TR1::placeholders::_1);
	moduleFactory_["LoadField"] = TR1::bind(&flow_control::LoadField::Create,
			this, TR1::placeholders::_1);

	moduleFactory_["Clock"] = TR1::bind(&flow_control::Clock::Create, this,
			TR1::placeholders::_1);
}
BaseContext::~BaseContext()
{
}

boost::optional<TR1::shared_ptr<Object> > BaseContext::FindObject(
		std::string const & name, std::type_info const &tinfo)
{
	TR1::shared_ptr<Object> res;

	if (name == "")
	{
		return boost::optional<TR1::shared_ptr<Object> >();
	}
	std::map<std::string, Object::Holder>::iterator it = objects.find(name);

	return boost::optional<TR1::shared_ptr<Object> >(
			it != objects.end()
					&& (tinfo == typeid(void) || it->second->CheckType(tinfo)),
			it->second);

}

boost::optional<TR1::shared_ptr<const Object> > BaseContext::FindObject(
		std::string const & name, std::type_info const &tinfo) const
{

	if (name == "")
	{
		return boost::optional<TR1::shared_ptr<const Object> >();
	}
	std::map<std::string, Object::Holder>::const_iterator it = objects.find(
			name);

	return boost::optional<TR1::shared_ptr<const Object> >(
			it != objects.end()
					&& (tinfo == typeid(void) || it->second->CheckType(tinfo)),
			it->second);

}
void BaseContext::DeleteObject(std::string const & name)
{
	typename std::map<std::string, TR1::shared_ptr<Object> >::iterator it =
			objects.find(name);
	if (it != objects.end())
	{
		objects.erase(it);
	}

}

void BaseContext::PushClock()
{
	timer_ += dt;
	++counter_;
}

void BaseContext::Load(ptree const & pt)
{
	eval_ = flow_control::Loop::Create(this, pt.get_child("Process"));
}

void BaseContext::Save()
{
}

void BaseContext::Eval()
{
	eval_();
}
//TR1::shared_ptr<BaseContext> BaseContext::Create(ptree const & pt)
//{
//	TR1::shared_ptr<BaseContext> res;
//
//	std::string topology = pt.get("Topology.<xmlattr>.Type", "CoRectMesh");
//
//	if (topology == "CoRectMesh")
//	{
//		res = TR1::shared_ptr<BaseContext>(new Context<UniformRectGrid>(pt))
//	}
//	else
//	{
//		ERROR << "Unregistered Context type!" << topology;
//	}
//
//	return res;
//
//}
}// namespace simpla

