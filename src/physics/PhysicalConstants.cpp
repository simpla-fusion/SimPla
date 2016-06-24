/*
 * physical_constants.cpp
 *
 *  created on: 2012-10-17
 *      Author: salmon
 */

#include "PhysicalConstants.h"

#include <iomanip>
#include "Constants.h"

namespace simpla
{
PhysicalConstants::PhysicalConstants(std::string type)
        : type_(type)
{
    SetBaseUnit(type_);
}

PhysicalConstants::~PhysicalConstants()
{
}

std::ostream &PhysicalConstants::save(std::ostream &os) const
{
    os

    << "{ "

    << "Type = \"" << type_ << "\" , "

    << "m = " << m_ << " , "

    << "s = " << s_ << " , "

    << "kg = " << kg_ << " , "

    << "C = " << C_ << " , "

    << "K  = " << K_ << " , "

    << "mol = " << mol_

    << "}";
    return os;
}

std::ostream &operator<<(std::ostream &os, PhysicalConstants const &self)
{
    return self.save(os);
}

void PhysicalConstants::SetBaseUnit(std::string const &type, Real pm, Real ps, Real pkg, Real pC, Real pK,
                                    Real pMol)
{
    type_ = type;

    if (type == "SI")
    {
        m_ = 1.0;
        s_ = 1.0;
        kg_ = 1.0;
        C_ = 1.0;
        K_ = 1.0;
        mol_ = 1.0;
    }
    else if (type == "CGS")
    {
        m_ = 100;
        s_ = 1.0;
        kg_ = 1000;
        C_ = 1.0;
        K_ = 1.0;
        mol_ = 1.0;
    }
    else if (type == "NATURE")
    {
        kg_ = 1.0 / SI_proton_mass;
        C_ = 1.0 / SI_elementary_charge;
        K_ = SI_elementary_charge / SI_Boltzmann_constant;
        m_ = C_ * C_ / (SI_permeability_of_free_space * kg_);
        s_ = m_ * SI_speed_of_light;
        mol_ = 1.0;
    }
    else
    {
        type_ = type;
        m_ = pm;
        s_ = ps;
        kg_ = pkg;
        C_ = pC;
        K_ = pK;
        mol_ = pMol;
    }

    q_["kg"] = kg_;

    q_["K"] = K_;

    q_["C"] = C_;

    q_["s"] = s_;

    q_["m"] = m_;

    q_["mol"] = mol_;

    q_["Hz"] = 1.0 / s_;

    q_["rad"] = 1.0;

    q_["sr"] = 1.0;

    q_["J"] = m_ * m_ * kg_ / s_ / s_; /* energy */

    q_["N"] = kg_ * m_ / s_ / s_; /* Force */

    q_["Pa"] = kg_ / m_ / s_ / s_; /*  Pressure */

    q_["W"] = kg_ * m_ * m_ / s_ / s_; /* Power    */

    q_["volt"] = kg_ * m_ * m_ / s_ / s_ / C_; /*Electric Potential  */

    q_["Ohm"] = kg_ * m_ * m_ / s_ / C_ / C_; /*ElectricResistance */

    q_["simens"] = s_ * C_ * C_ / kg_ / m_ / m_; /*ElectricConductance*/

    q_["F"] = s_ * s_ * C_ * C_ / kg_ / m_ / m_; /*Capacitance;    */

    q_["Wb"] = kg_ * m_ * m_ / s_ / C_; /* MagneticFlux    */

    q_["H"] = kg_ * m_ * m_ / C_ / C_; /*Magnetic inductance herny  */

    q_["Tesla"] = kg_ / s_ / C_; /*Magnetic induction   */

    q_["speed of light"] = SI_speed_of_light * m_ / s_; /*exact*/

    q_["permeability of free space"] = SI_permeability_of_free_space * q_["H"] / m_; /*exact*/

    q_["permittivity of free space"] = 1.0
                                       / (q_["speed of light"] * q_["speed of light"] *
                                          q_["permeability of free space"]);/*exact*/

    q_["mu"] = q_["permeability of free space"];

    q_["epsilon"] = q_["permittivity of free space"];

    q_["gravitational constant"] = SI_gravitational_constant * (m_ * m_ * m_) / (s_ * s_) / kg_; /*1.2e-4*/

    q_["plank constant"] = SI_plank_constant * q_["J"] * s_; /*4.4e-8*/

    q_["plank constant bar"] = SI_plank_constant_bar * q_["J"] * s_;

    q_["elementary charge"] = SI_elementary_charge * C_; /*2.2e-8*/

    q_["electron mass"] = SI_electron_mass * kg_; /*4.4e-8*/

    q_["proton mass"] = SI_proton_mass * kg_;

    q_["proton electron mass ratio"] = SI_proton_electron_mass_ratio;

    q_["electron charge mass ratio"] = SI_electron_charge_mass_ratio * C_ / kg_;

    q_["fine_structure constant"] = SI_fine_structure_constant; /*3.23-10*/

    q_["Rydberg constant"] = SI_Rydberg_constant / m_; /*5e-12*/

    q_["Avogadro constant"] = SI_Avogadro_constant / mol_; /*4.4e-8*/

    q_["Faraday constant"] = SI_Faraday_constant * C_ / mol_; /*2.2e-10*/

    q_["Boltzmann constant"] = SI_Boltzmann_constant * q_["J"] / K_; /*9.1e-7*/

    q_["electron volt"] = SI_electron_volt * q_["J"]; /*2.2e-8*/

    q_["atomic mass unit"] = SI_atomic_mass_unit * kg_; /*4.4e-8*/

}

std::ostream &PhysicalConstants::print(std::ostream &os) const
{
    os

    << "[Units] " << type_ << " ~ SI" << std::endl

    << std::setw(40) << "1 [length unit] = " << 1.0 / (*this)["m"] << "[m]" << std::endl

    << std::setw(40) << "1 [time unit] = " << 1.0 / (*this)["s"] << "[s]" << std::endl

    << std::setw(40) << "1 [mass unit] = " << 1.0 / (*this)["kg"] << "[kg]" << std::endl

    << std::setw(40) << "1 [electric charge unit] = "

    << 1.0 / (*this)["C"] << "[C]" << std::endl

    << std::setw(40) << "1 [temperature unit] = "

    << 1.0 / (*this)["K"] << "[K]" << std::endl

    << std::setw(40) << "1 [amount of substance] = "

    << 1.0 / (*this)["mol"] << "[mole]" << std::endl

    << "Physical constants:" << std::endl

    << std::setw(40) << "permeability of free space, mu = "

    << (*this)["permeability of free space"] << std::endl

    << std::setw(40) << "permittivity of free space, epsilon = "

    << (*this)["permittivity of free space"] << std::endl

    << std::setw(40) << "speed of light, c = "

    << (*this)["speed of light"] << std::endl

    << std::setw(40) << "elementary charge, e = "

    << (*this)["elementary charge"] << std::endl

    << std::setw(40) << "electron mass, m_e = "

    << (*this)["electron mass"] << std::endl

    << std::setw(40) << "proton mass,m_p = "

    << (*this)["proton mass"] << std::endl;

    return os;
}

}
// namespace simpla

