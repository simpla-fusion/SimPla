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
	void Config(TCONFIG const & vm)
	{
		SetBaseUnit(vm["type"].get_value<std::string>(),
				vm["m"].get_value<Real>(1.0), //
				vm["s"].get_value<Real>(1.0), //
				vm["kg"].get_value<Real>(1.0), //
				vm["C"].get_value<Real>(1.0), //
				vm["K"].get_value<Real>(1.0), //
				vm["Mol"].get_value<Real>(1.0));
	}

	std::string Summary() const;

	void SetBaseUnit(std::string const & type_name = "CUSTOM", Real pm = 1,
			Real ps = 1, Real pkg = 1, Real pC = 1, Real pK = 1, Real pMol = 1);

	inline Real operator[](std::string const &s) const
	{

		std::map<std::string, Real>::const_iterator it = q_.find(s);

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
	std::map<std::string, Real> q_; //physical quantity
	std::map<std::string, std::string> unitSymbol_;

	std::string type_;

//SI base unit
	Real m; //<< length [meter]
	Real s;	//<< time	[second]
	Real kg; //<< mass	[kilgram]
	Real C;	//<< electric charge	[coulomb]
	Real K;	//<< temperature [kelvin]
	Real mol;	//<< amount of substance [mole]

}
;
std::string Summary(PhysicalConstants const & self);

}  // namespace simpla

#endif /* PHYSICAL_CONSTANTS_H_ */
