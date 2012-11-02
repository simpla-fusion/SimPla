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
#include "modules/preprocess/preprocess.h"
namespace simpla
{

BaseContext::BaseContext() :
		dt(0.0), counter_(0), timer_(0), output_path("Untitled")
{
}
void BaseContext::Parse(ptree const&pt)
{
	dt = (pt.get("Grid.Time.<xmlattr>.dt", 1.0d)),

	PHYS_CONSTANTS.Parse(pt.get_child("PhysConstants"));

}
void BaseContext::Preprocess(ptree const&pt)
{
	preprocess::Preprocess(this, pt);
}
void BaseContext::Process(ptree const&pt)
{
	flow_control::Loop(*this, pt).Eval();
}
BaseContext::~BaseContext()
{
}

boost::optional<TR1::shared_ptr<Object> > BaseContext::FindObject(
		std::string const & name)
{
	if (name == "")
	{
		return boost::optional<TR1::shared_ptr<Object> >(false,
				TR1::shared_ptr<Object>());
	}

	std::map<std::string, TR1::shared_ptr<Object> >::iterator it = objects.find(
			name);

	return boost::optional<TR1::shared_ptr<Object> >(it != objects.end(),
			it->second);

}

boost::optional<TR1::shared_ptr<const Object> > BaseContext::FindObject(
		std::string const & name) const
{
	if (name == "")
	{
		return boost::optional<TR1::shared_ptr<const Object> >(false,
				TR1::shared_ptr<const Object>());
	}

	std::map<std::string, TR1::shared_ptr<Object> >::const_iterator it =
			objects.find(name);

	return boost::optional<TR1::shared_ptr<const Object> >(it != objects.end(),
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

void BaseContext::Save()
{
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

