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

std::string ContextList::add(std::string const & name,
		std::shared_ptr<Context> const & p)
{
	list_[name] = p;
	return "Context" + ToString(list_.size()) + "_" + name;
}

std::ostream & ContextList::print(std::ostream & os)
{
	for (auto const & item : list_)
	{
		os << item.first << std::endl;
	}
	return os;
}

void ContextList::setup(int argc, char const ** argv)
{
	for (auto const & item : list_)
	{

		LOGGER << "Context [" << item.first << "] setup ." << std::endl;

		item.second->setup(argc, argv);

		LOGGER << "Context [" << item.first << "] setup done." << std::endl;
	}
}

void ContextList::run()
{
	for (auto const & item : list_)
	{

		LOGGER << "Context [" << item.first << "] start." << std::endl;

		item.second->body();

		LOGGER << "Context [" << item.first << "] done." << std::endl;
	}
}


void ContextList::sync()
{
	for (auto const & item : list_)
	{

		LOGGER << "Context [" << item.first << "] start." << std::endl;

		item.second->sync();

		LOGGER << "Context [" << item.first << "] done." << std::endl;
	}
}
}  // namespace simpla

