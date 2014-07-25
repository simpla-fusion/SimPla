/**
 * \file properties.cpp
 *
 * \date    2014年7月13日  上午8:41:23 
 * \author salmon
 */

#include "properties.h"
namespace simpla
{
const Properties Properties::fail_safe_;
std::ostream & Properties::print(std::ostream & os) const
{
	for (auto const& item : *this)
	{
		os << item.first << " = " << item.second.value_ << " , ";
		if (item.second.size() > 0)
		{
			os << item.second;
		}

	}
	return os;
}

std::ostream & operator<<(std::ostream & os, Properties const & prop)
{
	return prop.print(os);
}
}  // namespace simpla
