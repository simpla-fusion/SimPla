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

		PHYS_CONSTANTS(pt.get_child_optional("PhysConstants")),

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
			it != objects.end()
					&& (tinfo == typeid(void) || it->second->CheckType(tinfo)),
			*it->second);

}

boost::optional<Object const&> BaseContext::FindObject(std::string const & name,
		std::type_info const &tinfo) const
{

	if (name == "")
	{
		return boost::optional<Object const &>();
	}
	std::map<std::string, Object::Holder>::const_iterator it = objects.find(
			name);

	return boost::optional<Object const &>(
			it != objects.end()
					&& (tinfo == typeid(void) || it->second->CheckType(tinfo)),
			*it->second);

}
void BaseContext::DeleteObject(std::string const & name)
{
	typename std::map<std::string, TR1::shared_ptr<Object> >::iterator it =
			objects.find(name);
	if (it != objects.end())
	{
		opool_.push_back(it->second);
		objects.erase(it);
	}

}

inline void eval_(TR1::function<void(void)> & f)
{
	f();
}
void BaseContext::Eval()
{

	LOG << "COUNTER:" << counter_;
	std::for_each(modules.begin(), modules.end(), eval_);
	++counter_;
	timer_ += dt;
}

}  // namespace simpla

