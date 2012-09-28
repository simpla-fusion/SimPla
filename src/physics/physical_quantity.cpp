/*
 * physical_quantity.cpp
 *
 *  Created on: 2012-8-3
 *      Author: salmon
 */
#include "physical_quantity.h"
//#include "constants.h"
#include <iostream>

namespace physics
{
namespace units
{
template<> const double UnitSystem<SI>::BaseUnits::Length = 1.0;
template<> const double UnitSystem<SI>::BaseUnits::Mass = 1.0;
template<> const double UnitSystem<SI>::BaseUnits::Time = 1.0;
template<> const double UnitSystem<SI>::BaseUnits::Charge = 1.0;
template<> const double UnitSystem<SI>::BaseUnits::Temperature = 1.0;
template<> const double UnitSystem<SI>::BaseUnits::Luminousintensity = 1.0;
template<> const double UnitSystem<SI>::BaseUnits::AmountOfSubstance = 1.0;

template<> const double UnitSystem<CGS>::BaseUnits::Length = 0.01;
template<> const double UnitSystem<CGS>::BaseUnits::Mass = 0.001;
template<> const double UnitSystem<CGS>::BaseUnits::Time = 1.0;
template<> const double UnitSystem<CGS>::BaseUnits::Charge = 1.0;
template<> const double UnitSystem<CGS>::BaseUnits::Temperature = 1.0;
template<> const double UnitSystem<CGS>::BaseUnits::Luminousintensity = 1.0;
template<> const double UnitSystem<CGS>::BaseUnits::AmountOfSubstance = 1.0;

} // namespace units
} // namespace physics

int main()
{
	using namespace physics::units;
	typedef UnitSystem<CGS> unit;
	unit a;
	std::cout << unit::Length::factor << std::endl //
			<< sizeof(a) << std::endl
//			<< m.value << std::endl //
//			<< g.value << std::endl //
//			<< s.value << std::endl //
//			<< C.value << std::endl //
//			<< K.value << std::endl
			;

}
