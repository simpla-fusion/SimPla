/*
 * physical_constants.h
 *
 *  Created on: 2012-10-17
 *      Author: salmon
 */

#ifndef PHYSICAL_CONSTANTS_H_
#define PHYSICAL_CONSTANTS_H_

#include <utilities/log.h>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>

namespace simpla
{

class PhysicalConstants
{
public:
	PhysicalConstants(std::string type = "SI");
	~PhysicalConstants();

	template<typename TCONFIG>
	void Config(TCONFIG vm)
	{

		SetBaseUnit(vm.template Get<std::string>("type"), //
		vm.template Get<double>("m", 1.0), //
		vm.template Get<double>("s", 1.0), //
		vm.template Get<double>("kg", 1.0), //
		vm.template Get<double>("C", 1.0), //
		vm.template Get<double>("K", 1.0), //
		vm.template Get<double>("Mol", 1.0));
	}

	std::string Summary() const;

	void SetBaseUnit(std::string const & type_name = "CUSTOM", double pm = 1,
			double ps = 1, double pkg = 1, double pC = 1, double pK = 1,
			double pMol = 1);

	inline double operator[](std::string const &s) const
	{

		std::map<std::string, double>::const_iterator it = q_.find(s);

		if (it != q_.end())
		{
			return it->second;
		}
		else
		{
			ERROR << "Physical quantity " << s << " is not available!";
		}

		return 0;
	}

private:
	std::map<std::string, double> q_; //physical quantity
	std::map<std::string, std::string> unitSymbol_;

	std::string type_;

//SI base unit
	double m; //<< length [meter]
	double s;	//<< time	[second]
	double kg; //<< mass	[kilgram]
	double C;	//<< electric charge	[coulomb]
	double K;	//<< temperature [kelvin]
	double mol;	//<< amount of substance [mole]

}
;
std::string Summary(PhysicalConstants const & self);

}  // namespace simpla

#endif /* PHYSICAL_CONSTANTS_H_ */
