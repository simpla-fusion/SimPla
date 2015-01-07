/*
 * context.cpp
 *
 *  Created on: 2015年1月2日
 *      Author: salmon
 */

#include "context.h"

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "../utilities/utilities.h"

namespace simpla
{

Context::Context(Context &, _impl::split)
{

}
Context::~Context()
{

}
void Context::setup(int argc, char const** argv)
{

}
virtual void Context::teardown()
{

}
Context Context::split()
{
	return Context(*this, _impl::split());
}

void Context::sync()
{

}
void Context::add_task(std::string const & name, std::shared_ptr<Task> const& p)
{
	task_list_.emplace_back(name, p);
}

void Context::body()
{
	for (auto & item : task_list_)
	{
		item.second->body(*this);
	}
}
}  // namespace simpla

