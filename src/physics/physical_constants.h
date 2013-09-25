/*
 * physical_constants.h
 *
 *  Created on: 2012-10-17
 *      Author: salmon
 */

#ifndef PHYSICAL_CONSTANTS_H_
#define PHYSICAL_CONSTANTS_H_
#include "include/simpla_defs.h"
#include "constants.h"
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

		SetBaseUnit(
		vm["type"].template as<std::string>(), //
				vm["m"].template as<double>(1.0), //
				vm["s"].template as<double>(1.0), //
				vm["kg"].template as<double>(1.0), //
				vm["C"].template as<double>(1.0), //
				vm["K"].template as<double>(1.0), //
				vm["Mol"].template as<double>(1.0));
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
