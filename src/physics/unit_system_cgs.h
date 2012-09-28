/*
 * unit_system_cgs.h
 *
 *  Created on: 2012-3-4
 *      Author: salmon
 */

#ifndef UNIT_SYSTEM_CGS_H_
#define UNIT_SYSTEM_CGS_H_
#include "physics/physics_quantity.h"
#include "physics/constants.h"
#include "physics/unit_system_inc.h"

namespace physics
{
namespace units
{

template<>
const std::string Symbol<CGS>::str[7]=
{ "cm", "g", "s", "C", "K", "cd", "mol" };

namespace cgs
{
//----------------------------------------------------------------------------------
// NRL Plasma Formulary
// p.10~12 Dimensions of frequently used physical quantities
//-----------------------     l  m  t  q  T  Iv n
typedef PhysicalQuantity<CGS, 1, 0, 0, 0, 0, 0, 0> Length;
typedef PhysicalQuantity<CGS, 0, 1, 0, 0, 0, 0, 0> Mass;
typedef PhysicalQuantity<CGS, 0, 0, 1, 0, 0, 0, 0> Time;
typedef PhysicalQuantity<CGS, 0, 0, 0, 1, 0, 0, 0> Charge;
typedef PhysicalQuantity<CGS, 0, 0, 0, 0, 1, 0, 0> Temperature;
typedef PhysicalQuantity<CGS, 0, 0, 0, 0, 0, 1, 0> LuminousIntensity;
typedef PhysicalQuantity<CGS, 0, 0, 0, 0, 0, 0, 1> AmountOfSubstance;

typedef double PlaneAngle;
typedef double SolidAngle;

static const Length m(100);
static const Mass g(1.0);
static const Time s(1.0);
static const Charge C(1.0);
static const Temperature K(1.0);
static const LuminousIntensity cd(1.0);
static const AmountOfSubstance mol(1.0);

typedef decltype(m/s) Velocity;
typedef decltype(C/s) Current;

static const auto A = C / s;

DECLARE_CONSTANTS_LIST

template<int I0, int I1, int I2, int I3, int I4, int I5, int I6>
PhysicalQuantity<SI, I0, I1, I2 + I3, I3, I4, I5, I6> ToSIUnit(
		PhysicalQuantity<CGS, I0, I1, I2, I3, I4, I5, I6> const & lhs)
{
	return (PhysicalQuantity<SI, I0, I1, I2 + I3, I3, I4, I5, I6>(
			lhs.value() / pow(m.value(), I0) / pow(kg.value(), I1)
					/ pow(s.value(), I2) / pow((A * s).value(), I3)
					/ pow(K.value(), I4) / pow(cd.value(), I5)
					/ pow(mol.value(), I6)));
}
;

template<int I0, int I1, int I2, int I3, int I4, int I5, int I6>
PhysicalQuantity<CGS, I0, I1, I2 - I3, I3, I4, I5, I6> FromSIUnit(
		PhysicalQuantity<SI, I0, I1, I2, I3, I4, I5, I6> const & lhs)
{
	return (PhysicalQuantity<CGS, I0, I1, I2 - I3, I3, I4, I5, I6>(
			lhs.value() * pow(m.value(), I0) * pow(kg.value(), I1)
					* pow(s.value(), I2) * pow((C / s).value(), I3)
					* pow(K.value(), I4) * pow(cd.value(), I5)
					* pow(mol.value(), I6)));
}
;

} // namespace cgs

} // namespace units

} // namespace physics

#endif /* UNIT_SYSTEM_CGS_H_ */
