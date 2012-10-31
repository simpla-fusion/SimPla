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
#include "constants.h"
namespace simpla
{

class PhysicalConstants
{
public:
	PhysicalConstants();
	~PhysicalConstants();

	void Parse(ptree const &pt);
	void Reset();

	std::string Summary() const;

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

}  // namespace simpla

#endif /* PHYSICAL_CONSTANTS_H_ */
