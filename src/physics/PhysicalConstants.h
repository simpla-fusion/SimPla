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
#include "../sp_config.h"
#include "../gtl/MemoryPool.h"
#include "Constants.h"

namespace simpla
{

/**
 * @ingroup physics
 * @addtogroup physical_constant  Physical constants and units
 */
/**
 * @ingroup physical_constant
 * @brief Physical constants and dimensions
 **/
class PhysicalConstants
{
public:

    typedef PhysicalConstants this_type;

    PhysicalConstants(std::string type = "SI");

    ~PhysicalConstants();

    friend std::ostream &operator<<(std::ostream &os,
                                    PhysicalConstants const &self);

    template<typename TDict>
    void load(TDict const &dict)
    {
        if (dict.empty())
        {
            SetBaseUnit();
        }
        else
        {

            SetBaseUnit(dict["Type"].template as<std::string>(), //
                        dict["m"].template as<Real>(1.0), //
                        dict["s"].template as<Real>(1.0), //
                        dict["kg"].template as<Real>(1.0), //
                        dict["C"].template as<Real>(1.0), //
                        dict["K"].template as<Real>(1.0), //
                        dict["mol"].template as<Real>(1.0));
        }
    }

    std::ostream &save(std::ostream &os) const;

    virtual std::ostream &print(std::ostream &os) const;

    void SetBaseUnit(std::string const &type_name = "SI", Real pm = 1,
                     Real ps = 1, Real pkg = 1, Real pC = 1, Real pK = 1,
                     Real pMol = 1);

    inline Real operator[](std::string const &s) const
    {
//
//		std::map<std::string, Real>::const_iterator it = q_.find(s);
//
//		if (it != q_.end())
//		{
//			return it->second;
//		}
//		else
//		{
//			THROW_EXCEPTION << "Physical quantity " << s << " is not available!";
//		}

        return q_.at(s);
    }

    std::string get_unit(std::string const &s) const
    {
        return unitSymbol_.at(s);

    }

private:
    std::map<std::string, Real> q_; //physical quantity
    std::map<std::string, std::string> unitSymbol_;

    std::string type_;

//SI PlaceHolder unit
    Real m_; //<< length [meter]
    Real s_;    //<< time	[second]
    Real kg_; //<< mass	[kilgram]
    Real C_;    //<< electric charge	[coulomb]
    Real K_;    //<< temperature [kelvin]
    Real mol_;    //<< amount of substance [mole]

};

std::ostream &operator<<(std::ostream &os, PhysicalConstants const &self);

#define  CONSTANTS    SingletonHolder<PhysicalConstants>::instance()

/**
 * @ingroup physical_constant
 * @brief Define physical constants:
 * @{
 */
#define DEFINE_PHYSICAL_CONST                                               \
const Real mu0 = CONSTANTS["permeability of free space"];                            \
const Real epsilon0 = CONSTANTS["permittivity of free space"];                       \
const Real speed_of_light = CONSTANTS["speed of light"];                             \
const Real speed_of_light2 =  speed_of_light*speed_of_light;                         \
const Real proton_mass = CONSTANTS["proton mass"];                                   \
const Real elementary_charge = CONSTANTS["elementary charge"];                       \
const Real boltzmann_constant = CONSTANTS["Boltzmann constant"];
//! @}

}// namespace simpla

#endif /* PHYSICAL_CONSTANTS_H_ */
