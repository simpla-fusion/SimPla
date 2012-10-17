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
namespace simpla
{
BaseContext::BaseContext(ptree const&pt) :
		dt(pt.get("dt", 1.0f)),

		PHYS_CONSTANTS(pt.get_child("PhysConstants")),

		counter_(0), timer_(0)

{

}
BaseContext::~BaseContext()
{
}

boost::optional<Object&> BaseContext::FindObject(std::string const & name,
		std::type_info const &tinfo)
{
	TR1::shared_ptr<Object> res;

	if (name == "")
	{
		return boost::optional<Object &>();
	}
	std::map<std::string, Object::Holder>::iterator it = objects.find(name);

	return boost::optional<Object &>(
			it != objects.end() && it->second->CheckType(tinfo), *it->second);

}

void BaseContext::DeleteObject(std::string const & name)
{
	typename std::map<std::string, TR1::shared_ptr<Object> >::iterator it =
			objects.find(name);
	if (it != objects.end())
	{
		unnamed_objects.push_back(it->second);
		objects.erase(it);
	}

}

inline void eval_(TR1::function<void(void)> & f)
{
	f();
}
void BaseContext::Eval()
{
	++counter_;
	std::for_each(modules.begin(), modules.end(), eval_);
}

}  // namespace simpla

