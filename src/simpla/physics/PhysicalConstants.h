/**
 * @file PhysicalConstants.h
 *
 *  created on: 2012-10-17
 *      Author: salmon
 */

#ifndef PHYSICAL_CONSTANTS_H_
#define PHYSICAL_CONSTANTS_H_

#include <iostream>
#include <map>
#include <string>
#include "simpla/utilities/Constants.h"

namespace simpla {

/**
 * @ingroup physics
 * @addtogroup physical_constant  Physical constants and units
 */
/**
 * @ingroup physical_constant
 * @brief Physical constants and dimensions
 **/
class PhysicalConstants {
   public:
    typedef PhysicalConstants this_type;

    PhysicalConstants(std::string type = "SI");

    ~PhysicalConstants();

    friend std::ostream &operator<<(std::ostream &os, PhysicalConstants const &self);

    template <typename TDict>
    void load(TDict const &dict) {
        if (dict.empty()) {
            SetBaseUnit();
        } else {
            SetBaseUnit(dict["Type"].template as<std::string>(),  //
                        dict["m"].template as<double>(1.0),       //
                        dict["s"].template as<double>(1.0),       //
                        dict["kg"].template as<double>(1.0),      //
                        dict["C"].template as<double>(1.0),       //
                        dict["K"].template as<double>(1.0),       //
                        dict["mol"].template as<double>(1.0));
        }
    }

    std::ostream &save(std::ostream &os) const;

    virtual std::ostream &print(std::ostream &os) const;

    void SetBaseUnit(std::string const &type_name = "SI", double pm = 1, double ps = 1, double pkg = 1, double pC = 1,
                     double pK = 1, double pMol = 1);

    inline double operator[](std::string const &s) const {
        //
        //		std::map<std::string, double>::const_iterator it = q_.find(s);
        //
        //		if (it != q_.end())
        //		{
        //			return it->m_node_;
        //		}
        //		else
        //		{
        //			THROW_EXCEPTION << "Physical quantity " << s << " is not available!";
        //		}

        return q_.at(s);
    }

    std::string get_unit(std::string const &s) const { return unitSymbol_.at(s); }

   private:
    std::map<std::string, double> q_;  // physical quantity
    std::map<std::string, std::string> unitSymbol_;

    std::string type_;

    // SI PlaceHolder unit
    double m_;    //<< length [meter]
    double s_;    //<< time	[m_node_]
    double kg_;   //<< mass	[kilgram]
    double C_;    //<< electric charge	[coulomb]
    double K_;    //<< temperature [kelvin]
    double mol_;  //<< amount of substance [mole]
};

std::ostream &operator<<(std::ostream &os, PhysicalConstants const &self);

#define CONSTANTS SingletonHolder<PhysicalConstants>::instance()

/**
 * @ingroup physical_constant
 * @brief Define physical constants:
 * @{
 */
#define DEFINE_PHYSICAL_CONST                                        \
    const double mu0 = CONSTANTS["permeability of free space"];      \
    const double epsilon0 = CONSTANTS["permittivity of free space"]; \
    const double speed_of_light = CONSTANTS["speed of light"];       \
    const double speed_of_light2 = speed_of_light * speed_of_light;  \
    const double proton_mass = CONSTANTS["proton mass"];             \
    const double elementary_charge = CONSTANTS["elementary charge"]; \
    const double boltzmann_constant = CONSTANTS["Boltzmann constant"];
//! @}

}  // namespace simpla

#endif /* PHYSICAL_CONSTANTS_H_ */
