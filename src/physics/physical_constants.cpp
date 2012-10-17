/*
 * physical_constants.cpp
 *
 *  Created on: 2012-10-17
 *      Author: salmon
 */
#include "physical_constants.h"
namespace simpla
{

PhysicalConstants::PhysicalConstants(boost::optional<ptree const &> opt)
{
	if (!opt || !opt->get_optional<std::string>("<xmlattr>.type"))
	{
		type = "SI";
	}
	else
	{
		type = opt->get<std::string>("<xmlattr>.type");
		if (type == "CUSTOM")
		{
			m = opt->get("m", 1.0f);
			s = opt->get("s", 1.0f);
			kg = opt->get("kg", 1.0f);
			C = opt->get("C", 1.0f);
			K = opt->get("K", 1.0f);
			mol = opt->get("mol", 1.0f);
		}
	}

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

#define METRIC_PRREFIXES(_NAME_)
	q_["Hz"] = 1.0 / s;
	METRIC_PRREFIXES (Hz)

	q_["rad"] = 1.0;
	METRIC_PRREFIXES (rad)

	q_["sr"] = 1.0;
	METRIC_PRREFIXES (sr)

	q_["J"] = m * m * kg / s / s; /* energy */
	METRIC_PRREFIXES (J)

	q_["N"] = kg * m / s / s; /* Force */
	METRIC_PRREFIXES (N)

	q_["Pa"] = kg / m / s / s; /*  Pressure */
	METRIC_PRREFIXES (Pa)

	q_["W"] = kg * m * m / s / s; /* Power    */
	METRIC_PRREFIXES (W)

	q_["volt"] = kg * m * m / s / s / C; /*Electric Potential  */
	METRIC_PRREFIXES (volt)

	q_["Ohm"] = kg * m * m / s / C / C; /*ElectricResistance */
	METRIC_PRREFIXES (Ohm)

	q_["simens"] = s * C * C / kg / m / m; /*ElectricConductance*/
	METRIC_PRREFIXES (simens)

	q_["F"] = s * s * C * C / kg / m / m; /*Capacitance;    */
	METRIC_PRREFIXES (F)

	q_["Wb"] = kg * m * m / s / C; /* MagneticFlux    */
	METRIC_PRREFIXES (Wb)

	q_["H"] = kg * m * m / C / C; /*Magnetic inductance herny  */
	METRIC_PRREFIXES (H)

	q_["Tesla"] = kg / s / C; /*Magnetic induction   */
	METRIC_PRREFIXES (Tesla)

#undef METRIC_PRREFIXES

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
PhysicalConstants::~PhysicalConstants()
{
	;
}
std::string PhysicalConstants::Summary() const
{
	std::ostringstream os;

	os

	<< DOUBLELINE << std::endl

	<< "Units " << type << "~ SI" << std::endl

	<< SINGLELINE << std::endl

	<< std::setw(40) << "1 [length unit] = " << 1.0 / m << "[m]" << std::endl

	<< std::setw(40) << "1 [time unit] = " << 1.0 / s << "[s]" << std::endl

	<< std::setw(40) << "1 [mass unit] = " << 1.0 / kg << "[kg]" << std::endl

	<< std::setw(40) << "1 [electric charge unit] = "

	<< 1.0 / C << "[C]" << std::endl

	<< std::setw(40) << "1 [temperature unit] = "

	<< 1.0 / K << "[K]" << std::endl

	<< std::setw(40) << "1 [amount of substance] = "

	<< 1.0 / mol << "[mole]" << std::endl

	<< SINGLELINE << std::endl

	<< "Physical constants:" << std::endl

	<< SINGLELINE << std::endl

	<< std::setw(40) << "permeability of free space, mu = "

	<< (*this)["permeability_of_free_space"] << std::endl

	<< std::setw(40) << "permittivity of free space, epsilon = "

	<< (*this)["permittivity_of_free_space"] << std::endl

	<< std::setw(40) << "speed of light, c = "

	<< (*this)["speed_of_light"] << std::endl

	<< std::setw(40) << "elementary charge, e = "

	<< (*this)["elementary_charge"] << std::endl

	<< std::setw(40) << "electron mass, m_e = "

	<< (*this)["electron_mass"] << std::endl

	<< std::setw(40) << "proton mass,m_p = "

	<< (*this)["proton_mass"] << std::endl

	<< DOUBLELINE << std::endl;

	return os.str();

}

}  // namespace simpla

