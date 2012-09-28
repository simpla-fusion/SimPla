/*
 * physical_quantity.h
 *
 *  Created on: 2012-8-3
 *      Author: salmon
 */

#ifndef PHYSICAL_QUANTITY_H_
#define PHYSICAL_QUANTITY_H_
#include <cmath>
#include "primitives/operation.h"
namespace simpla
{
namespace physics
{

template<int I0, int I1, int I2, int I3, int I4, int I5, int I6,
		typename BASE_UNIT, typename TV = double>
class PhysicalQuantity
{
public:
	typedef TV ValueType;
	static const double factor;
	ValueType value;

	PhysicalQuantity(ValueType v) :
			value(v)
	{
	}
	~PhysicalQuantity()
	{

	}

};

template<int I0, int I1, int I2, int I3, int I4, int I5, int I6,
		typename BASE_UNIT, typename TV>
const double PhysicalQuantity<I0, I1, I2, I3, I4, I5, I6, BASE_UNIT, TV>::factor =
		pow(BASE_UNIT::Length, I0) //
		* pow(BASE_UNIT::Mass, I1) //
				* pow(BASE_UNIT::Time, I2) //
				* pow(BASE_UNIT::Charge, I3) //
				* pow(BASE_UNIT::Temperature, I4) //
				* pow(BASE_UNIT::Luminousintensity, I5) //
				* pow(BASE_UNIT::AmountOfSubstance, I6);

template<typename TL> struct PhysicalQuantityTraits
{
	typedef TL ValueType;
	inline ValueType value(ValueType const & v)
	{
		return v;
	}
};

template<typename UL, typename TL>
struct PhysicalQuantityTraits<PhysicalQuantity<0, 0, 0, 0, 0, 0, UL, TL> >
{
	typedef TL ValueType;
	inline ValueType value(PhysicalQuantity<0, 0, 0, 0, 0, 0, UL, TL> const & v)
	{
		return v.value;
	}
};

template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename UL,
		typename TL, int R0, int R1, int R2, int R3, int R4, int R5, int R6,
		typename UR, typename TR>
inline PhysicalQuantity<L0 + R0, L1 + R1, L2 + R2, L3 + R3, L4 + R4, L5 + R5,
		L6 + R6, UL,
		typename TypeOpTraits<TL, TR, arithmetic::OpMultiplication>::ValueType>  //
operator*(PhysicalQuantity<L0, L1, L2, L3, L4, L5, L6, UL, TL> const & lhs,
		PhysicalQuantity<R0, R1, R2, R3, R4, R5, R6, UR, TR> const & rhs)
{
	typedef PhysicalQuantity<L0 + R0, L1 + R1, L2 + R2, L3 + R3, L4 + R4,
			L5 + R5, L6 + R6, UL,
			typename TypeOpTraits<TL, TR, arithmetic::OpMultiplication>::ValueType> ResultType;
	return ((lhs.value * lhs.facotr * rhs.value * rhs.factor)
			/ ResultType::factor);
}

template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TL, int R0, int R1, int R2, int R3, int R4, int R5, int R6,
		typename TR>
inline typename PhysicalQuantityTraits<
		PhysicalQuantity<US, L0 - R0, L1 - R1, L2 - R2, L3 - R3, L4 - R4,
				L5 - R5, L6 - R6, decltype(TL()/TR())> >::ValueType //
operator/(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TL> const & lhs,
		PhysicalQuantity<US, R0, R1, R2, R3, R4, R5, R6, TR> const & rhs)
{
	return (PhysicalQuantityTraits<
			PhysicalQuantity<US, L0 - R0, L1 - R1, L2 - R2, L3 - R3, L4 - R4,
					L5 - R5, L6 - R6, decltype(TL()/TR())>>::get_value(
			lhs.value() / rhs.value()));
}

template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TV>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV>                   //
operator+(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV> const & lhs,
		PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV> const & rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV>(
			lhs.value() + rhs.value()));
}
template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TV>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV>                   //
operator-(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV> const & lhs,
		PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV> const & rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV>(
			lhs.value() - rhs.value()));
}
template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TL, typename TR>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, decltype(TL()*TR())>  //
operator*(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TL> const & lhs,
		TR const &rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, decltype(TL()*TR())>(
			lhs.value() * rhs));
}
template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TV>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV>                   //
operator*(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV> const & lhs,
		TV const &rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV>(
			lhs.value() * rhs));
}
template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TR>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TR>                   //
operator*(TR const & lhs,
		PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TR> const &rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TR>(
			lhs * rhs.value()));
}
template<int US, typename TL, int L0, int L1, int L2, int L3, int L4, int L5,
		int L6, typename TR>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, decltype(TL()*TR())>  //
operator*(TL const & lhs,
		PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TR> const &rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, decltype(TL()*TR())>(
			lhs * rhs.value()));
}

template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TL, typename TR>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, decltype(TL(),TR())>  //
operator/(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TL> const & lhs,
		TR rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, decltype(TL(),TR())>(
			lhs.value() / rhs));
}
template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TL>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TL>                   //
operator/(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TL> const & lhs,
		TL const & rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TL>(
			lhs.value() / rhs));
}

template<int US, typename TL, int L0, int L1, int L2, int L3, int L4, int L5,
		int L6, typename TR>
inline PhysicalQuantity<US, -L0, -L1, -L2, -L3, -L4, -L5, -L6,
		decltype(TL()/TR())>                                 //
operator/(TL lhs,
		PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TR> const & rhs)
{
	return (PhysicalQuantity<US, -L0, -L1, -L2, -L3, -L4, -L5, -L6,
			decltype(TL()/TR())>(lhs / rhs.value()));
}

template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TR>
inline PhysicalQuantity<US, -L0, -L1, -L2, -L3, -L4, -L5, -L6, TR>            //
operator/(TR lhs,
		PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TR> const & rhs)
{
	return (PhysicalQuantity<US, -L0, -L1, -L2, -L3, -L4, -L5, -L6, TR>(
			lhs / rhs.value()));
}

namespace units
{

#define METRIC_PRREFIXES(_TYPE_,_NAME_)                                     \
 const _TYPE_ P##_NAME_ = 1.0e15 *  _NAME_;                           \
 const _TYPE_ T##_NAME_ = 1.0e12 *  _NAME_;                           \
 const _TYPE_ G##_NAME_ = 1.0e9 *  _NAME_;                            \
 const _TYPE_ M##_NAME_ = 1.0e6 *  _NAME_;                            \
 const _TYPE_ k##_NAME_ = 1.0e3 *  _NAME_;                            \
 const _TYPE_ c##_NAME_ = 1.0e-2 * _NAME_;                            \
 const _TYPE_ m##_NAME_ = 1.0e-3 * _NAME_;                            \
 const _TYPE_ u##_NAME_ = 1.0e-6 * _NAME_;                            \
 const _TYPE_ n##_NAME_ = 1.0e-9 * _NAME_;                            \
 const _TYPE_ p##_NAME_ = 1.0e-12 * _NAME_;                           \
 const _TYPE_ f##_NAME_ = 1.0e-15 * _NAME_;

enum
{
	SI, CGS, NATURE, GAUSSIAN
};

template<int IUS>
struct UnitSystem
{

	struct BaseUnits
	{
		static const double Length;
		static const double Mass;
		static const double Time;
		static const double Charge;
		static const double Temperature;
		static const double Luminousintensity;
		static const double AmountOfSubstance;
	};
//-------------------------- l  m  t  q  T  Iv n
	typedef PhysicalQuantity<0, 0, 0, 0, 0, 0, 0, BaseUnits, double> DIMESIONLESS;
	typedef PhysicalQuantity<1, 0, 0, 0, 0, 0, 0, BaseUnits, double> Length;
	typedef PhysicalQuantity<0, 1, 0, 0, 0, 0, 0, BaseUnits, double> Mass;
	typedef PhysicalQuantity<0, 0, 1, 0, 0, 0, 0, BaseUnits, double> Time;
	typedef PhysicalQuantity<0, 0, 0, 1, 0, 0, 0, BaseUnits, double> Charge;
	typedef PhysicalQuantity<0, 0, 0, 0, 1, 0, 0, BaseUnits, double> Temperature;
	typedef PhysicalQuantity<0, 0, 0, 0, 0, 1, 0, BaseUnits, double> Luminousintensity;
	typedef PhysicalQuantity<0, 0, 0, 0, 0, 0, 1, BaseUnits, double> AmountOfSubstance;
	typedef PhysicalQuantity<0, 2, 0, 0, 0, 0, 0, BaseUnits, double> Area;
	typedef PhysicalQuantity<-2, -1, 2, 2, 0, 0, 0, BaseUnits, double> Capacitance;
	typedef PhysicalQuantity<-3, 0, 0, 1, 0, 0, 0, BaseUnits, double> ChargeDensity;
	typedef PhysicalQuantity<-2, -1, 1, 2, 0, 0, 0, BaseUnits, double> ElectricConductance;
	typedef PhysicalQuantity<-3, -1, 1, 2, 0, 0, 0, BaseUnits, double> Conductivity;
	typedef PhysicalQuantity<0, 0, -1, 1, 0, 0, 0, BaseUnits, double> CurrentDensity;
	typedef PhysicalQuantity<-3, 0, 0, 0, 0, 0, 0, BaseUnits, double> Density;
	typedef PhysicalQuantity<-2, 0, 0, 1, 0, 0, 0, BaseUnits, double> Displacement;
	typedef PhysicalQuantity<1, 1, -2, -1, 0, 0, 0, BaseUnits, double> ElectricField;
	typedef PhysicalQuantity<2, 1, -2, -1, 0, 0, 0, BaseUnits, double> Electromotance;
	typedef PhysicalQuantity<2, 1, -2, 0, 0, 0, 0, BaseUnits, double> Energy;
	typedef PhysicalQuantity<-1, 1, -2, 0, 0, 0, 0, BaseUnits, double> EnergyDensity;
	typedef PhysicalQuantity<1, 1, -2, 0, 0, 0, 0, BaseUnits, double> Force;
	typedef PhysicalQuantity<0, 0, -1, 0, 0, 0, 0, BaseUnits, double> Frequency;
	typedef PhysicalQuantity<2, 1, -1, -2, 0, 0, 0, BaseUnits, double> Impedance;
	typedef PhysicalQuantity<2, 1, 0, -2, 0, 0, 0, BaseUnits, double> Inductance;
	typedef PhysicalQuantity<-1, 0, -1, 1, 0, 0, 0, BaseUnits, double> MagneticIntensity;
	typedef PhysicalQuantity<2, 1, -1, -1, 0, 0, 0, BaseUnits, double> MagneticFlux;
	typedef PhysicalQuantity<0, 1, -1, -1, 0, 0, 0, BaseUnits, double> MagneticInduction;
	typedef PhysicalQuantity<2, 0, -1, 1, 0, 0, 0, BaseUnits, double> MagneticMoment;
	typedef PhysicalQuantity<-1, 0, -1, 1, 0, 0, 0, BaseUnits, double> Magnetization;
	typedef PhysicalQuantity<0, 0, -1, 1, 0, 0, 0, BaseUnits, double> Magnetomotance;
	typedef PhysicalQuantity<1, 1, -1, 0, 0, 0, 0, BaseUnits, double> Momentum;
	typedef PhysicalQuantity<-2, 1, -1, 0, 0, 0, 0, BaseUnits, double> MomentumDensity;
	typedef PhysicalQuantity<1, 1, 0, -2, 0, 0, 0, BaseUnits, double> Permeability;
	typedef PhysicalQuantity<-3, -1, 2, 2, 0, 0, 0, BaseUnits, double> Permittivity;
	typedef PhysicalQuantity<-2, 0, 0, 1, 0, 0, 0, BaseUnits, double> Polarization;
	typedef PhysicalQuantity<2, 1, -2, -1, 0, 0, 0, BaseUnits, double> Potential;
	typedef PhysicalQuantity<2, 1, -3, 0, 0, 0, 0, BaseUnits, double> Power;
	typedef PhysicalQuantity<-1, 1, -3, 0, 0, 0, 0, BaseUnits, double> PowerDensity;
	typedef PhysicalQuantity<-1, 1, -2, 0, 0, 0, 0, BaseUnits, double> Pressure;
	typedef PhysicalQuantity<-2, -1, 0, 2, 0, 0, 0, BaseUnits, double> Reluctance;
	typedef PhysicalQuantity<2, 1, -1, -2, 0, 0, 0, BaseUnits, double> Resistance;
	typedef PhysicalQuantity<3, 1, -1, -2, 0, 0, 0, BaseUnits, double> Resistivity;
	typedef PhysicalQuantity<1, 1, -3, 0, 0, 0, 0, BaseUnits, double> ThermalConductivity;
	typedef PhysicalQuantity<1, 1, -1, -1, 0, 0, 0, BaseUnits, double> VectorPotential;
	typedef PhysicalQuantity<-1, 1, -1, 0, 0, 0, 0, BaseUnits, double> Viscosity;
	typedef PhysicalQuantity<3, 0, 0, 0, 0, 0, 0, BaseUnits, double> Volume;
	typedef PhysicalQuantity<0, 0, -1, 0, 0, 0, 0, BaseUnits, double> Vorticity;
	typedef PhysicalQuantity<2, 1, -2, 0, 0, 0, 0, BaseUnits, double> Work;

};

//const UnitSystem<SI>::Length m(1.0);
//METRIC_PRREFIXES(UnitSystem<SI>::Length, m)
//const UnitSystem<SI>::Mass g(1.0e-3);
//METRIC_PRREFIXES(UnitSystem<SI>::Mass, g)
//const UnitSystem<SI>::Time s(1.0);
//METRIC_PRREFIXES(UnitSystem<SI>::Time, s)
//const UnitSystem<SI>::Charge C(1.0);
//METRIC_PRREFIXES(UnitSystem<SI>::Charge, C)
//const UnitSystem<SI>::Temperature K(1.0);
//METRIC_PRREFIXES(UnitSystem<SI>::Temperature, K)

}// namespace units
}  //namespace physics
} //namespace simpla
#endif /* PHYsiCAL_QUANTITY_H_ */
