/*
 * physical_constants.h
 *
 *  Created on: 2012-10-17
 *      Author: salmon
 */

#ifndef PHYSICAL_CONSTANTS_H_
#define PHYSICAL_CONSTANTS_H_

#include "../utilities/log.h"
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
	void Deserialize(TCONFIG vm)
	{

		SetBaseUnit(vm.template Get<std::string>("Type"), //
		vm.template Get<double>("m", 1.0), //
		vm.template Get<double>("s", 1.0), //
		vm.template Get<double>("kg", 1.0), //
		vm.template Get<double>("C", 1.0), //
		vm.template Get<double>("K", 1.0), //
		vm.template Get<double>("mol", 1.0));
	}

	template<typename TCONFIG>
	void Serialize(TCONFIG vm)
	{

		vm.template SetValue<std::string>("Type", type_);
		vm.template SetValue<double>("m", m_);
		vm.template SetValue<double>("s", s_);
		vm.template SetValue<double>("kg", kg_);
		vm.template SetValue<double>("C", C_);
		vm.template SetValue<double>("K", K_);
		vm.template SetValue<double>("mol", mol_);
	}

	inline std::basic_ostream<char> & Serialize(
			std::basic_ostream<char> &os) const
	{
		os

		<< SINGLELINE << std::endl

		<< "--# Unit System" << std::endl

		<< "Type" << "\t= " << type_ << std::endl

		<< "m" << "\t= " << m_ << std::endl

		<< "s" << "\t= " << s_ << std::endl

		<< "kg" << "\t= " << kg_ << std::endl

		<< "C" << "\t= " << C_ << std::endl

		<< "K" << "\t= " << K_ << std::endl

		<< "mol" << "\t= " << mol_ << std::endl

		<< "--" << SINGLELINE << std::endl;
		return os;
	}

	void Print(std::basic_ostream<char> & os) const;

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
	double m_; //<< length [meter]
	double s_;	//<< time	[second]
	double kg_; //<< mass	[kilgram]
	double C_;	//<< electric charge	[coulomb]
	double K_;	//<< temperature [kelvin]
	double mol_;	//<< amount of substance [mole]

}
;

}  // namespace simpla

#endif /* PHYSICAL_CONSTANTS_H_ */
