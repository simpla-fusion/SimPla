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
#include "grid/grid.h"
#include "modules/modules.h"
#include "utilities/properties.h"

namespace simpla
{

BaseContext::BaseContext() :
		dt(0.0), counter_(0), timer_(0)
{
	objFactory_["Compound"] = std::bind(&CompoundObject::Create, this,
			std::placeholders::_1);

}
void BaseContext::Parse(PTree const&pt)
{
	dt = (pt.get("Grid.Time.<xmlattr>.dt", 1.0d)),

	PHYS_CONSTANTS.Parse(pt.get_child("PhysConstants"));

}

void BaseContext::Process(PTree const&pt)
{
	flow_control::Loop(this, pt).Eval();
}
BaseContext::~BaseContext()

{
}

//std::shared_ptr<BaseContext> BaseContext::Create(ptree const & pt)
//{
//	std::shared_ptr<BaseContext> res;
//
//	std::string topology = pt.get("Topology.<xmlattr>.Type", "CoRectMesh");
//
//	if (topology == "CoRectMesh")
//	{
//		res = std::shared_ptr<BaseContext>(new Context<UniformRectGrid>(pt))
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

