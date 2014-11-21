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
void UseCase::parse_cmd_line(
		std::function<int(std::string const &, std::string const &)> const & fun)
{
	simpla::parse_cmd_line(argc_, argv_, fun);
}

std::tuple<bool, std::string> UseCase::cmdline_option(
		std::string const & name) const
{
	return find_option_from_cmd_line(argc_, argv_, name);
}
void UseCase::run(int argc, char ** argv)
{
	argc_ = argc;
	argv_ = argv;

	bool is_configure_test_ = false;

	simpla::parse_cmd_line(argc_, argv_,
			[&](std::string const & opt,std::string const & value)->int
			{
				if(opt=="i"||opt=="input")
				{
					dict_.parse_file(value);
				}
				else if(opt=="e"|| opt=="execute")
				{
					dict_.parse_string(value);
				}
				else if (opt=="h"|| opt=="help")
				{

					SHOW_OPTIONS("-i,--input <STRING>","input configure file");
					SHOW_OPTIONS("-e,--execute <STRING>","execute Lua script as configuration");

					return TERMINATE;
				}
				return CONTINUE;

			}

			);

	if (!is_configure_test_)
	{
//		INFORM << "Use case:" << case_info_ << " begin" << std::endl;
		case_body();
//		INFORM << "Use case:" << case_info_ << " end" << std::endl;

	}

}

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
		item.second->run(argc, argv);
	}
}
}  // namespace simpla
