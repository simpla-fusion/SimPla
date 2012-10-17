/*
 * grid.cpp
 *
 *  Created on: 2012-10-7
 *      Author: salmon
 */

#include "grid.h"
namespace simpla
{

std::map<std::string, TR1::function<TR1::shared_ptr<BaseGrid>(ptree const & pt)> > BaseGrid::callback_;

TR1::shared_ptr<BaseGrid> BaseGrid::GridFactory(ptree const & pt)
{
	std::string type = pt.get("Type", "UniformRect");
	std::map<std::string,
			TR1::function<TR1::shared_ptr<BaseGrid>(ptree const & pt)> >::iterator it =
			callback_.find(type);
	if (it == callback_.end())
	{
		ERROR << "Unknown grid type \"" << type << "\"!";
	}
	return ((it->second)(pt));
}

}  // namespace simpla
