/*
 * physical_constants.h
 *
 *  Created on: 2012-10-17
 *      Author: salmon
 */

#ifndef PHYSICAL_CONSTANTS_H_
#define PHYSICAL_CONSTANTS_H_

#include "../utilities/log.h"
#include "../utilities/lua_state.h"
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

	typedef PhysicalConstants this_type;

	PhysicalConstants(std::string type = "SI");

	~PhysicalConstants();

	friend std::ostream & operator<<(std::ostream &os, PhysicalConstants const &self);

	void Load(LuaObject const & vm);

	std::ostream & Save(std::ostream & os)const;

	void Print(std::basic_ostream<char> & os) const;

	void SetBaseUnit(std::string const & type_name = "SI", double pm = 1, double ps = 1, double pkg = 1, double pC = 1,
	        double pK = 1, double pMol = 1);

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
	double m_; //<< length [meter]
	double s_;	//<< time	[second]
	double kg_; //<< mass	[kilgram]
	double C_;	//<< electric charge	[coulomb]
	double K_;	//<< temperature [kelvin]
	double mol_;	//<< amount of substance [mole]

}
;
std::ostream & operator<<(std::ostream & os, PhysicalConstants const & self);

#define DEFINE_PHYSICAL_CONST(_UNIT_SYS_)                                               \
PhysicalConstants const & CONSTANTS=_UNIT_SYS_;                                         \
const double mu0 = CONSTANTS["permeability of free space"];                            \
const double epsilon0 = CONSTANTS["permittivity of free space"];                       \
const double speed_of_light = CONSTANTS["speed of light"];                             \
const double speed_of_light2 =  speed_of_light*speed_of_light;                         \
const double proton_mass = CONSTANTS["proton mass"];                                   \
const double elementary_charge = CONSTANTS["elementary charge"];                       \
const double boltzmann_constant = CONSTANTS["Boltzmann constant"];

#define GLOBAL_PHYSICAL_CONST    SingletonHolder<PhysicalConstants>::instance()
#define DEFINE_GLOBAL_PHYSICAL_CONST   DEFINE_PHYSICAL_CONST(SingletonHolder<PhysicalConstants>::instance())
}  // namespace simpla

#endif /* PHYSICAL_CONSTANTS_H_ */
