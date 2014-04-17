// ----------------------------------------------------------------------------
// Copyright (C) 2002-2006 Marcin Kalicinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// For more information, see www.boost.org
// ----------------------------------------------------------------------------

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <string>
#include <set>
#include <exception>
#include <iostream>
#include "include/simpla_defs.h"
#include "primitives/ntuple.h"
#include "primitives/properties.h"
#include <boost/foreach.hpp>

using namespace simpla;

int main(int argc, char ** argv)
{
	try
	{
		using boost::property_tree::ptree;
		ptree pt;
		read_file(argv[1], pt);
		std::cout

		<< "Name:" << pt.get<std::string>("name") << std::endl

		<< "dt:" << pt.get<Real>("dt") << std::endl << "dx:"

		<< pt.get<std::string>("dx") << std::endl << "dx:"

		<< pt.get<Vec3>("dx", pt_trans<Vec3, typename ptree::data_type>())
				<< std::endl

				<< pt.get<IVec3>("dims",
						pt_trans<IVec3, typename ptree::data_type>())
				<< std::endl;

		std::pair<ptree::assoc_iterator, ptree::assoc_iterator> it_range =
				pt.equal_range("particles");
		for (typename ptree::assoc_iterator it = it_range.first;
				it != it_range.second; ++it)
		{
			std::cout<< it->first<< it->second.get< std::string > ("name") << std::endl;
		}

		write_file(argv[1] + std::string(".") + argv[2], pt);
	} catch (std::exception &e)
	{
		std::cout << "Error: " << e.what() << "\n";
	}
	return 0;
}
