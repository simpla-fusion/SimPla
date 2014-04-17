/*
 * unit_system_si.h
 *
 *  Created on: 2012-3-4
 *      Author: salmon
 */

#ifndef UNIT_SYSTEM_SI_H_
#define UNIT_SYSTEM_SI_H_
#include "physics/physics_quantity.h"
#include "physics/constants.h"
#include "physics/unit_system_inc.h"

namespace physics
{
namespace units
{
//----------------------------------------------------------------------------------
// NRL Plasma Formulary
// p.10~12 Dimensions of frequently used physical quantities
// p.13 International system (SI) nomenclature
template<>
const std::string Symbol<SI>::str[7]=
{ "m", "kg", "s", "A", "K", "cd", "mol" };
namespace si
{
//-------------------------- l  m  t  I  T  Iv n
typedef PhysicalQuantity<SI, 1, 0, 0, 0, 0, 0, 0> Length;
typedef PhysicalQuantity<SI, 0, 1, 0, 0, 0, 0, 0> Mass;
typedef PhysicalQuantity<SI, 0, 0, 1, 0, 0, 0, 0> Time;
typedef PhysicalQuantity<SI, 0, 0, 0, 1, 0, 0, 0> Current;
typedef PhysicalQuantity<SI, 0, 0, 0, 0, 1, 0, 0> Temperature;
typedef PhysicalQuantity<SI, 0, 0, 0, 0, 0, 1, 0> LuminousIntensity;
typedef PhysicalQuantity<SI, 0, 0, 0, 0, 0, 0, 1> AmountOfSubstance;

typedef double PlaneAngle;
typedef double SolidAngle;

static const Length m(1.0);
static const Mass g(1.0e-3);
static const Time s(1.0);
static const Current A(1.0);
static const Temperature K(1.0);
static const LuminousIntensity cd(1.0);
static const AmountOfSubstance mol(1.0);

typedef decltype(m/s) Velocity;
typedef decltype(A*s) Charge;
static const auto C = A * s;

DECLARE_CONSTANTS_LIST

template<int I0, int I1, int I2, int I3, int I4, int I5, int I6>
PhysicalQuantity<SI, I0, I1, I2, I3, I4, I5, I6> ToSIUnit(
		PhysicalQuantity<SI, I0, I1, I2, I3, I4, I5, I6> const & lhs)
{
	return (lhs);
}

template<int I0, int I1, int I2, int I3, int I4, int I5, int I6>
PhysicalQuantity<SI, I0, I1, I2, I3, I4, I5, I6> FromSIUnit(
		PhysicalQuantity<SI, I0, I1, I2, I3, I4, I5, I6> const & lhs)
{
	return (lhs);
}
} // namespace SI

} // namespace units

} // namespace physics

#endif /* UNIT_SYSTEM_SI_H_ */
