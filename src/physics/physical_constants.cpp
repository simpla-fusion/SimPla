/*
 * physical_constants.cpp
 *
 *  Created on: 2012-10-17
 *      Author: salmon
 */
#include "physical_constants.h"
namespace simpla
{
PhysicalConstants::PhysicalConstants() :
		type("SI")
{
	Init();
}
PhysicalConstants::~PhysicalConstants()
{
}

void PhysicalConstants::Init()
{
	if (type == "SI")
	{
		m = 1.0;
		s = 1.0;
		kg = 1.0;
		C = 1.0;
		K = 1.0;
		mol = 1.0;
	}
	else if (type == "CGS")
	{
		m = 100;
		s = 1.0;
		kg = 1000;
		C = 1.0;
		K = 1.0;
		mol = 1.0;
	}
	else if (type == "NATURE")
	{
		kg = 1.0 / SI_proton_mass;
		C = 1.0 / SI_elementary_charge;
		K = SI_elementary_charge / SI_Boltzmann_constant;
		m = C * C / (SI_permeability_of_free_space * kg);
		s = m * SI_speed_of_light;
		mol = 1.0;
	}

	q_["kg"] = kg;

	q_["K"] = K;

	q_["C"] = C;

	q_["s"] = s;

	q_["m"] = m;

	q_["mol"] = mol;

	q_["Hz"] = 1.0 / s;

	q_["rad"] = 1.0;

	q_["sr"] = 1.0;

	q_["J"] = m * m * kg / s / s; /* energy */

	q_["N"] = kg * m / s / s; /* Force */

	q_["Pa"] = kg / m / s / s; /*  Pressure */

	q_["W"] = kg * m * m / s / s; /* Power    */

	q_["volt"] = kg * m * m / s / s / C; /*Electric Potential  */

	q_["Ohm"] = kg * m * m / s / C / C; /*ElectricResistance */

	q_["simens"] = s * C * C / kg / m / m; /*ElectricConductance*/

	q_["F"] = s * s * C * C / kg / m / m; /*Capacitance;    */

	q_["Wb"] = kg * m * m / s / C; /* MagneticFlux    */

	q_["H"] = kg * m * m / C / C; /*Magnetic inductance herny  */

	q_["Tesla"] = kg / s / C; /*Magnetic induction   */

	q_["speed_of_light"] = SI_speed_of_light * m / s; /*exact*/

	q_["permeability_of_free_space"] = SI_permeability_of_free_space * q_["H"]
			/ m; /*exact*/

	q_["permittivity_of_free_space"] = 1.0
			/ (q_["speed_of_light"] * q_["speed_of_light"]
					* q_["permeability_of_free_space"]);/*exact*/

	q_["mu"] = q_["permeability_of_free_space"];

	q_["epsilon"] = q_["permittivity_of_free_space"];

	q_["gravitational_constant"] = SI_gravitational_constant * (m * m * m)
			/ (s * s) / kg; /*1.2e-4*/

	q_["plank_constant"] = SI_plank_constant * q_["J"] * s; /*4.4e-8*/

	q_["plank_constant_bar"] = SI_plank_constant_bar * q_["J"] * s;

	q_["elementary_charge"] = SI_elementary_charge * C; /*2.2e-8*/

	q_["electron_mass"] = SI_electron_mass * kg; /*4.4e-8*/

	q_["proton_mass"] = SI_proton_mass * kg;

	q_["proton_electron_mass_ratio"] = SI_proton_electron_mass_ratio;

	q_["electron_charge_mass_ratio"] = SI_electron_charge_mass_ratio * C / kg;

	q_["fine_structure_constant"] = SI_fine_structure_constant; /*3.23-10*/

	q_["Rydberg_constant"] = SI_Rydberg_constant / m; /*5e-12*/

	q_["Avogadro_constant"] = SI_Avogadro_constant / mol; /*4.4e-8*/

	q_["Faraday_constant"] = SI_Faraday_constant * C / mol; /*2.2e-10*/

	q_["Boltzmann_constant"] = SI_Boltzmann_constant * q_["J"] / K; /*9.1e-7*/

	q_["electron_volt"] = SI_electron_volt * q_["J"]; /*2.2e-8*/

	q_["atomic_mass_unit"] = SI_atomic_mass_unit * kg; /*4.4e-8*/

}


std::string Summary(PhysicalConstants* self)
{
	std::ostringstream os;

	os

	<< DOUBLELINE << std::endl

	<< "Units " << self->type << " ~ SI" << std::endl

	<< SINGLELINE << std::endl

	<< std::setw(40) << "1 [length unit] = " << 1.0 / (*self)["m"] << "[m]"
			<< std::endl

			<< std::setw(40) << "1 [time unit] = " << 1.0 / (*self)["s"]
			<< "[s]" << std::endl

			<< std::setw(40) << "1 [mass unit] = " << 1.0 / (*self)["kg"]
			<< "[kg]" << std::endl

			<< std::setw(40) << "1 [electric charge unit] = "

			<< 1.0 / (*self)["C"] << "[C]" << std::endl

			<< std::setw(40) << "1 [temperature unit] = "

			<< 1.0 / (*self)["K"] << "[K]" << std::endl

			<< std::setw(40) << "1 [amount of substance] = "

			<< 1.0 / (*self)["mol"] << "[mole]" << std::endl

			<< SINGLELINE << std::endl

			<< "Physical constants:" << std::endl

			<< SINGLELINE << std::endl

			<< std::setw(40) << "permeability of free space, mu = "

			<< (*self)["permeability_of_free_space"] << std::endl

			<< std::setw(40) << "permittivity of free space, epsilon = "

			<< (*self)["permittivity_of_free_space"] << std::endl

			<< std::setw(40) << "speed of light, c = "

			<< (*self)["speed_of_light"] << std::endl

			<< std::setw(40) << "elementary charge, e = "

			<< (*self)["elementary_charge"] << std::endl

			<< std::setw(40) << "electron mass, m_e = "

			<< (*self)["electron_mass"] << std::endl

			<< std::setw(40) << "proton mass,m_p = "

			<< (*self)["proton_mass"] << std::endl

			<< DOUBLELINE << std::endl;

	return os.str();

}


}  // namespace simpla

