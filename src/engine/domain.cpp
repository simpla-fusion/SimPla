/*
 * domain.cpp
 *
 *  Created on: 2012-10-13
 *      Author: salmon
 */
#include <iostream>
#include "include/simpla_defs.h"
#include "domain.h"

namespace simpla
{

;
std::string Domain::Summary()
{
	std::ostringstream os;

	os << "[Domain ]" << std::endl

	<< "[Grid]" << std::endl

	<< grid_->Summary() << std::endl

	<< SINGLELINE << std::endl

	<< DOUBLELINE << std::endl;
//	std::cout << desc << std::endl;
//
//	std::cout << SINGLELINE << std::endl;
//
//	std::cout << SINGLELINE << std::endl;
//	std::cout << "[Unit & Dimensions]" << std::endl;
//
//	std::cout << std::setw(20) << "Unit Dimensions : " << unit_dimensions
//			<< std::endl;
//
//	std::cout << SINGLELINE << std::endl;
//

//
//	std::cout << "[Boundary Condition]" << std::endl
//			<< " (< 0 for CYCLE, 0 for PEC, 1 for Mur ABC ,>1 for PML ABC)"
//			<< std::endl;
//
//	std::cout << std::setw(20) << "LEFT : " << bc[0] << std::endl;
//
//	std::cout << std::setw(20) << "RIGHT : " << bc[1] << std::endl;
//
//	std::cout << std::setw(20) << "FRONT : " << bc[2] << std::endl;
//
//	std::cout << std::setw(20) << "BACK : " << bc[3] << std::endl;
//
//	std::cout << std::setw(20) << "UP : " << bc[4] << std::endl;
//
//	std::cout << std::setw(20) << "DOWN : " << bc[5] << std::endl;
//
//	std::cout << SINGLELINE << std::endl;
//
//	std::cout << "[ Functions List]" << std::endl;
//
//	std::cout << SINGLELINE << std::endl;
//
//	std::cout << std::endl //
//			<< SINGLELINE << std::endl //
//			<< "[Predefine Species]" << std::endl //
//			<< SINGLELINE << std::endl //
//			<< std::setw(20) << " Name  | " //
//			<< " Description" << std::endl;
//	for (Context::SpeciesMap::iterator it = species_.begin();
//			it != species_.end(); ++it)
//	{
//		std::cout
//
//		<< std::setw(17) << it->first << " | "
//
//		<< " q/e = " << boost::any_cast<double>(it->second["Z"])
//
//		<< ", m/m_p = " << boost::any_cast<double>(it->second["m"])
//
//		<< ", T = " << boost::any_cast<double>(it->second["T"]) << "[eV]"
//
//		<< ", pic = " << boost::any_cast<double>(it->second["pic"])
//
//		<< std::endl
//
//		<< std::setw(20) << " | " << ", engine = "
//
//		<< boost::any_cast<std::string>(it->second["engine"])
//
//		<< std::endl;
	return os.str();
}

}  // namespace simpla

