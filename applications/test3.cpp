/*
 * test3.cpp
 *
 *  Created on: 2013年11月7日
 *      Author: salmon
 */

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <iostream>

int main(int argc, char** argv)
{
	using boost::property_tree::ptree;
//	ptree pt;
//	ptree pt_grid;
//
//	pt_grid.put("<xmlattr>.Name", "Test");
//
//	pt.put_child("Grid", pt_grid);
//
//	boost::property_tree::write_xml("test.xml", pt, std::locale());
//
	ptree pt2;

	boost::property_tree::read_xml(argv[1], pt2);

	auto it_range = pt2.get_child("Xdmf.Domain.Grid.").equal_range("Attribute");

	for (auto it = it_range.first; it != it_range.second; ++it)
	{
		std::cout << it->second.get<std::string>("<xmlattr>.Name") << std::endl;
	}

}
