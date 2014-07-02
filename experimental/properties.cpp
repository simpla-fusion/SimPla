/*
 * properties.cpp
 *
 *  Created on: 2012-10-7
 *      Author: salmon
 */

#include "properties.h"
#include <boost/property_tree/xml_parser.hpp>

namespace simpla
{

void read_file(std::string const & fname, PTree & pt)
{

	boost::property_tree::read_xml(fname,
			reinterpret_cast<boost::property_tree::ptree &>(pt));

}

void write_file(std::string const & fname, PTree const & pt)
{
	boost::property_tree::write_xml(fname,
			reinterpret_cast<boost::property_tree::ptree const&>(pt));

}
} //namespace simpla
