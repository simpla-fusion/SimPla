/*
 * physical_constants.h
 *
 *  Created on: 2012-3-3
 *      Author: salmon
 */
#ifndef PHYSICAL_CONSTANTS_H_
#define PHYSICAL_CONSTANTS_H_
namespace physics
{
namespace units
{
enum Tag
{
	DIMENSION_LESS, SI, CGS, NATURE, CUSTOM
};
template<int US> struct Symbol
{
	static const std::string str[7];
};
} // namespace units
} // namespace physics

#define METRIC_PRREFIXES(_NAME_)                                     \
static const auto P##_NAME_ = 1.0e15 *  _NAME_;                           \
static const auto T##_NAME_ = 1.0e12 *  _NAME_;                           \
static const auto G##_NAME_ = 1.0e9 *  _NAME_;                            \
static const auto M##_NAME_ = 1.0e6 *  _NAME_;                            \
static const auto k##_NAME_ = 1.0e3 *  _NAME_;                            \
static const auto c##_NAME_ = 1.0e-2 * _NAME_;                            \
static const auto m##_NAME_ = 1.0e-3 * _NAME_;                            \
static const auto u##_NAME_ = 1.0e-6 * _NAME_;                            \
static const auto n##_NAME_ = 1.0e-9 * _NAME_;                            \
static const auto p##_NAME_ = 1.0e-12 * _NAME_;                           \
static const auto f##_NAME_ = 1.0e-15 * _NAME_;                           \


#define DECLARE_CONSTANTS_LIST                                                                \
typedef decltype(m*m) Area;                                                                   \
typedef decltype(s*s*C*C/g/m/m) Capacitance;                                                 \
typedef decltype(C/m/m/m) ChargeDensity;                                                      \
typedef decltype(C*C*s/m/m/g) ElectricConductance;                                           \
typedef decltype(C*C*s/g/m/m/m) Conductivity;                                                \
																							  \
typedef decltype(C/s/m/m) CurrentDensity;                                                     \
typedef decltype(1/m/m/m) Density;                                                            \
typedef decltype(C/m/m) Displacement;                                                         \
																							  \
typedef decltype(m*g/s/s/C ) ElectricField;                                                  \
typedef decltype(m*m*g/s/s/C ) ElectricPotential;                                            \
typedef decltype(m*m*g/s/C/C) ElectricResistance;                                            \
typedef decltype(m*m*m*g/s/C/C) ElectricResistivity;                                         \
																							  \
typedef decltype(m*m*g/s/s) Energy;                                                          \
typedef decltype(g/m/s/s) EnergyDensity;                                                     \
typedef decltype(m*g/s/s) Force;                                                             \
typedef decltype(1/s) Frequency;                                                              \
typedef decltype(m*m*g/s/C/C) Impedance;                                                     \
typedef decltype(m*m*g/C/C) MagneticInductance;                                              \
typedef decltype(C/m/s) MagneticIntensity;                                                    \
typedef decltype(m*m*g/s/C) MagneticFlux;                                                    \
typedef decltype(g/s/C) MagneticInduction;                                                   \
typedef decltype(m*m/s) MagneticMoment;                                                       \
typedef decltype(C/m/s) Magnetization;                                                        \
typedef decltype(m*g/s) Momentum;                                                            \
typedef decltype(m*g/C/C) Permeability;                                                      \
typedef decltype(C*C*s*s/m/m/m/g) Permittivity;                                              \
typedef decltype(C/m/m) Polarization;                                                         \
																							  \
typedef decltype(m*m*g/s/s/s) Power;                                                         \
typedef decltype(g/m/s/s/s) PowerDensity;                                                    \
typedef decltype(g/m/s/s) Pressure;                                                          \
																							  \
typedef decltype(m*g/s/s/s) ThermalConductivity;                                             \
typedef decltype(m*g/s/C) VectorPotential;                                                   \
typedef decltype(g/m/s) Viscosity;                                                           \
typedef decltype(m*m*m) Volume;                                                               \
typedef decltype(1/s) Vorticity;                                                              \
typedef decltype(m*m*g/s/s) Work;                                                            \
                                                                                              \
METRIC_PRREFIXES( m)                                                                          \
METRIC_PRREFIXES( g)                                                                          \
METRIC_PRREFIXES( s)                                                                          \
METRIC_PRREFIXES( C)                                                                          \
METRIC_PRREFIXES( K)                                                                          \
METRIC_PRREFIXES( cd)                                                                         \
METRIC_PRREFIXES( mol)                                                                        \
                                                                                              \
static const auto Hz = 1.0 / s;                                                               \
METRIC_PRREFIXES( Hz)                                                                         \
                                                                                              \
static const auto rad = 1.0;                                                                  \
METRIC_PRREFIXES( rad)                                                                        \
                                                                                              \
static const auto sr = 1.0;                                                                   \
METRIC_PRREFIXES( sr)                                                                         \
                                                                                              \
static const auto J = m * m * kg / s / s; /* energy */                                        \
METRIC_PRREFIXES( J)                                                                          \
                                                                                              \
static const auto N = kg * m / s / s; /* Force */                                             \
METRIC_PRREFIXES( N)                                                                          \
                                                                                              \
static const auto Pa = kg / m / s / s; /*  Pressure */                                        \
METRIC_PRREFIXES( Pa)                                                                         \
                                                                                              \
static const auto W = kg * m * m / s / s; /* Power    */                                      \
METRIC_PRREFIXES( W)                                                                          \
                                                                                              \
static const auto volt = kg * m * m / s / s / C; /*Electric Potential  */                     \
METRIC_PRREFIXES( volt)                                                                       \
                                                                                              \
static const auto Ohm = kg * m * m / s / C / C; /*ElectricResistance */                       \
METRIC_PRREFIXES( Ohm)                                                                        \
                                                                                              \
static const auto simens = s * C * C / kg / m / m; /*ElectricConductance*/                    \
METRIC_PRREFIXES( simens)                                                                     \
                                                                                              \
static const auto F = s * s * C * C / kg / m / m; /*Capacitance;    */                        \
METRIC_PRREFIXES( F)                                                                          \
                                                                                              \
static const auto Wb = kg * m * m / s / C; /* MagneticFlux    */                              \
METRIC_PRREFIXES( Wb)                                                                         \
                                                                                              \
static const auto H = kg * m * m / C / C; /*Magnetic inductance herny  */                     \
METRIC_PRREFIXES( H)                                                                          \
                                                                                              \
static const auto Tesla = kg / s / C; /*Magnetic induction   */                               \
METRIC_PRREFIXES( Tesla)                                                                      \
                                                                                              \
                                                                                              \
/* Fundamental Physical Constants — Frequently used constants */                                       \
/* Ref: http://physics.nist.gov/cuu/Constants/ */                                                      \
static const auto speed_of_light				= SI_speed_of_light * m / s;       /*exact*/                 \
static const auto permeability_of_free_space 	= SI_permeability_of_free_space * H / m; /*exact*/                       \
static const auto permittivity_of_free_space 	=                                                      \
	        1.0/(speed_of_light*speed_of_light*permeability_of_free_space);/*exact*/                   \
static const auto gravitational_constant		= SI_gravitational_constant * (m * m * m) / (s * s)/ kg; /*1.2e-4*/  \
static const auto plank_constant				= SI_plank_constant * J * s;    /*4.4e-8*/                \
static const auto plank_constant_bar 			= SI_plank_constant_bar * J * s;                             \
static const auto elementary_charge 			= SI_elementary_charge * C;     /*2.2e-8*/                   \
static const auto electron_mass					= SI_electron_mass * kg;         /*4.4e-8*/              \
static const auto proton_mass 					= SI_proton_mass * kg;                                \
static const auto proton_electron_mass_ratio 	= SI_proton_electron_mass_ratio;                                       \
static const auto electron_charge_mass_ratio 	= SI_electron_charge_mass_ratio * C / kg;                                  \
static const auto fine_structure_constant		= SI_fine_structure_constant; /*3.23-10*/                         \
static const auto Rydberg_constant 				= SI_Rydberg_constant/m; /*5e-12*/                         \
static const auto Avogadro_constant 			= SI_Avogadro_constant/mol; /*4.4e-8*/                        \
static const auto Faraday_constant				= SI_Faraday_constant*C/mol;  /*2.2e-10*/                       \
static const auto Boltzmann_constant 			= SI_Boltzmann_constant * J / K;   /*9.1e-7*/                  \
static const auto electron_volt					= SI_electron_volt*J; /*2.2e-8*/                        \
static const auto atomic_mass_unit				= SI_atomic_mass_unit*kg; /*4.4e-8*/                       \
///* Fundamental Physical Constants — Frequently used constants */                                       \
///* Ref: http://physics.nist.gov/cuu/Constants/ */                                                      \
//static const auto speed_of_light				= 299792458.0 * m / s;       /*exact*/                 \
//static const auto permeability_of_free_space 	= 4.0e-7 * PI * H / m; /*exact*/                       \
//static const auto permittivity_of_free_space 	=                                                      \
//	        1.0/(speed_of_light*speed_of_light*permeability_of_free_space);/*exact*/                   \
//static const auto gravitational_constant		= 6.67384e-11 * (m * m * m) / (s * s)/ kg; /*1.2e-4*/  \
//static const auto plank_constant				= 6.62606957e-34 * J * s;    /*4.4e-8*/                \
//static const auto plank_constant_bar 			= 1.054571726e-34 * J * s;                             \
//static const auto elementary_charge 			= 1.60217656e-19 * C;     /*2.2e-8*/                   \
//static const auto electron_mass					= 9.10938291e-31 * kg;         /*4.4e-8*/              \
//static const auto proton_mass 					= 1.672621777e-27 * kg;                                \
//static const auto proton_electron_mass_ratio 	= 1836.15267245;                                       \
//static const auto electron_charge_mass_ratio 	= 1.7588e11 * C / kg;                                  \
//static const auto fine_structure_constant		= 7.2973525698e-3; /*3.23-10*/                         \
//static const auto Rydberg_constant 				= 10973731.568539/m; /*5e-12*/                         \
//static const auto Avogadro_constant 			= 6.02214129e23/mol; /*4.4e-8*/                        \
//static const auto Faraday_constant				= 96485.3365*C/mol;  /*2.2e-10*/                       \
//static const auto Boltzmann_constant 			= 1.3806488e-23 * J / K;   /*9.1e-7*/                  \
//static const auto electron_volt					= 1.602176565e-19*J; /*2.2e-8*/                        \
//static const auto atomic_mass_unit				= 1.660538921e-27*kg; /*4.4e-8*/                       \

//1.0/(speed_of_light*speed_of_light*permeability_of_free_space);
//#undef METRIC_PRREFIXES
#endif // PHYSICAL_CONSTANTS_H_
