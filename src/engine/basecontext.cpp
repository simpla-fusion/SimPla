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
#include "modules/modules.h"

namespace simpla
{

BaseContext::BaseContext() :
		dt(0.0), counter_(0), timer_(0)
{
	objFactory_["Compound"] = TR1::bind(&CompoundObject::Create, this,
			TR1::placeholders::_1);

}
void BaseContext::Parse(ptree const&pt)
{
	dt = (pt.get("Grid.Time.<xmlattr>.dt", 1.0d)),

	PHYS_CONSTANTS.Parse(pt.get_child("PhysConstants"));

}

void BaseContext::Process(ptree const&pt)
{
	flow_control::Loop(this, pt).Eval();
}
BaseContext::~BaseContext()

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

