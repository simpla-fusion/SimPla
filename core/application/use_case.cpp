/*
 * use_case.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include "use_case.h"

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "../simpla_defs.h"
#include "../utilities/lua_state.h"
#include "../utilities/parse_command_line.h"

namespace simpla
{

std::string UseCaseList::add(std::string const & name,
		std::shared_ptr<UseCase> const & p)
{
	list_[name] = p;
	return "UseCase" + ToString(list_.size()) + "_" + name;
}

std::ostream & UseCaseList::print(std::ostream & os)
{
	for (auto const & item : list_)
	{
		os << item.first << std::endl;
	}
	return os;
}

void UseCaseList::run_all_case(int argc, char ** argv)
{
	for (auto const & item : list_)
	{
		item.second->init(argc, argv);
		item.second->body();
	}
}
}  // namespace simpla
