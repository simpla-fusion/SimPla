/*
 * physical_constants.h
 *
 *  Created on: 2012-10-17
 *      Author: salmon
 */

#ifndef PHYSICAL_CONSTANTS_H_
#define PHYSICAL_CONSTANTS_H_
#include "include/simpla_defs.h"
#include "utilities/properties.h"
#include "primitives/primitives.h"
#include "constants.h"
namespace simpla
{

class PhysicalConstants
{
public:
	PhysicalConstants();
	~PhysicalConstants();

	void Parse(ptree const &pt);

	void Init();

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

	inline void SetBaseUnits(std::string const & type_name, Real pm, Real ps,
			Real pkg, Real pC, Real pK, Real pMol)
	{
		type = type_name;
		m = pm;
		s = ps;
		kg = pkg;
		C = pC;
		K = pK;
		mol = pMol;
		Init();
	}

private:
	std::map<std::string, Real> q_; //physical quantity
	std::map<std::string, std::string> unitSymbol_;

	std::string type;

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
