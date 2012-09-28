/*
 * unit_system_cgs.h
 *
 *  Created on: 2012-3-4
 *      Author: salmon
 */

#ifndef UNIT_SYSTEM_NATURE_H_
#define UNIT_SYSTEM_NATURE_H_
#include "physics/physics_quantity.h"
#include "physics/constants.h"
#include "physics/unit_system_inc.h"
namespace physics
{
namespace units
{

template<>
const std::string Symbol<NATURE>::str[7]=
{ "c", "mp", "s", "e", "eV", "cd", "mol" };
// let mp=e=c=\epsilon_0=\mu_0=k=1

namespace nature
{
//----------------------------------------------------------------------------------
// NRL Plasma Formulary
//-----------------------        v  m  t  q  T  Iv n
typedef PhysicalQuantity<NATURE, 1, 0, 0, 0, 0, 0, 0, double> Velocity;
typedef PhysicalQuantity<NATURE, 0, 1, 0, 0, 0, 0, 0, double> Mass;
typedef PhysicalQuantity<NATURE, 0, 0, 1, 0, 0, 0, 0, double> Time;
typedef PhysicalQuantity<NATURE, 0, 0, 0, 1, 0, 0, 0, double> Charge;
typedef PhysicalQuantity<NATURE, 0, 0, 0, 0, 1, 0, 0, double> Temperature;
typedef PhysicalQuantity<NATURE, 0, 0, 0, 0, 0, 1, 0, double> LuminousIntensity;
typedef PhysicalQuantity<NATURE, 0, 0, 0, 0, 0, 0, 1, double> AmountOfSubstance;

typedef PhysicalQuantity<NATURE, 1, 0, 1, 0, 0, 0, 0, double> Length;
typedef PhysicalQuantity<NATURE, 0, 0, -1, 1, 0, 0, 0, double> Current;

static const LuminousIntensity cd(1.0);
static const AmountOfSubstance mol(1.0);

static const Mass mp(1.0);
static const Charge e(1.0);
static const Temperature eV(1.0);

static const Mass g(1.0 / SI_atomic_mass_unit);
static const Charge C(1.0 / SI_elementary_charge);
static const Temperature K(SI_elementary_charge / SI_Boltzmann_constant);
static const Length m(
		(C * C / (SI_permeability_of_free_space * 1000.0 * g)).value());
static const Time s(SI_speed_of_light * m.value());
static const Current A(C / s);

DECLARE_CONSTANTS_LIST

template<int I0, int I1, int I2, int I3, int I4, int I5, int I6>
PhysicalQuantity<SI, I0, I1, I2 - I0 + I3, I3, I4, I5, I6> ToSIUnit(
		PhysicalQuantity<NATURE, I0, I1, I2, I3, I4, I5, I6> const & lhs)
{
	return (PhysicalQuantity<SI, I0, I1, I2 - I0 + I3, I3, I4, I5, I6>(
			lhs.value()

			/ pow((nature::m / nature::s).value(), I0)

			/ pow(nature::kg.value(), I1)

			/ pow(nature::s.value(), I2)

			/ pow((nature::A * nature::s).value(), I3)

			/ pow(nature::K.value(), I4)

			/ pow(nature::cd.value(), I5) / pow(nature::mol.value(), I6)));
}

template<int I0, int I1, int I2, int I3, int I4, int I5, int I6>
PhysicalQuantity<NATURE, I0, I1, I2 + I0 - I3, I3, I4, I5, I6> FromSIUnit(
		PhysicalQuantity<SI, I0, I1, I2, I3, I4, I5, I6> const & lhs)
{
	return (PhysicalQuantity<NATURE, I0, I1, I2 + I0 - I3, I3, I4, I5, I6>(
			lhs.value() *

			pow(nature::m.value(), I0) *

			pow(nature::kg.value(), I1) *

			pow(nature::s.value(), I2) *

			pow((nature::C / nature::s).value(), I3) *

			pow(nature::K.value(), I4) *

			pow(nature::cd.value(), I5) *

			pow(nature::mol.value(), I6)));
}

} // namespace nature
} // namespace units
} // namespace physics

#endif /* UNIT_SYSTEM_NATURE_H_ */
