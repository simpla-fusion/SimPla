/*
 * dimensioned_quantity.h
 *
 *  Created on: 2012-3-18
 *      Author: salmon
 */

#ifndef DIMENSIONED_QUANTITY_H_
#define DIMENSIONED_QUANTITY_H_
#include <type_traits>
#include <utility>
#include "primitives/expression.h"
namespace simpla
{

template<int I0, int I1, int I2, int I3, int I4, int I5, int I6, typename TV>
class DimensionedQuantity
{
public:
	typedef TV Value;
	ReferenceTraits<Value> value;

	DimensionedQuantity(Value v) :
			value(v)
	{
	}
	~DimensionedQuantity()
	{

	}

};

template<typename T>
struct DimensionLessQuantityTraits
{
	typedef typename

	std::conditional<
			std::is_same<
					DimensionedQuantity<0, 0, 0, 0, 0, 0, 0, typename T::Value>,
					T>::value, typename T::Value, T>::type type;
};
template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TL,
		int R0, int R1, int R2, int R3, int R4, int R5, int R6, typename TR>
inline auto operator*(
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TL> const & lhs,
		DimensionedQuantity<R0, R1, R2, R3, R4, R5, R6, TR> const & rhs)
		->typename DimensionLessQuantityTraits<
		DimensionedQuantity<L0 + R0, L1 + R1, L2 + R2, L3 + R3, L4 + R4,
		L5 + R5, L6 + R6, decltype(lhs.value*rhs.value)>
		>::type
{
	return (typename DimensionLessQuantityTraits<
			DimensionedQuantity<L0 + R0, L1 + R1, L2 + R2, L3 + R3, L4 + R4,
					L5 + R5, L6 + R6, decltype(lhs.value*rhs.value)> >::type(
			lhs.value * rhs.value));
}

template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TL,
		int R0, int R1, int R2, int R3, int R4, int R5, int R6, typename TR>
inline auto operator/(
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TL> const & lhs,
		DimensionedQuantity<R0, R1, R2, R3, R4, R5, R6, TR> const & rhs)
		->typename DimensionLessQuantityTraits<
		DimensionedQuantity<L0 - R0, L1 - R1, L2 - R2, L3 - R3, L4 - R4,
		L5 - R5, L6 - R6, decltype(lhs.value/rhs.value) > >::type
{
	return (typename DimensionLessQuantityTraits<
			DimensionedQuantity<L0 - R0, L1 - R1, L2 - R2, L3 - R3, L4 - R4,
					L5 - R5, L6 - R6, decltype(lhs.value/rhs.value)> >::type(
			lhs.value / rhs.value));
}

template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TV>
inline auto operator+(
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TV> const & lhs,
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TV> const & rhs)
				DECL_RET_TYPE(
						(DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6,
								decltype(lhs.value + rhs.value)>( lhs.value + rhs.value ))

				)

template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TV>
inline auto operator-(
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TV> const & lhs,
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TV> const & rhs)
				DECL_RET_TYPE(
						(DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6,
								decltype(lhs.value - rhs.value)>( lhs.value - rhs.value ))

				)

template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TL,
		typename TR>
inline auto operator*(
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TL> const & lhs,
		TR const &rhs)
				DECL_RET_TYPE(
						(
								DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, decltype(lhs.value*rhs)>(
										lhs.value * rhs)
						))

template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TL,
		typename TR>
inline auto operator*(TL const &lhs,
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TR> const & rhs,
		)
				DECL_RET_TYPE(
						(
								DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, decltype(lhs*rhs.value)>(
										lhs * rhs.value)
						))
template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TL,
		typename TR>
inline auto operator/(
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TL> const & lhs,
		TR const &rhs)
				DECL_RET_TYPE(
						(
								DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, decltype(lhs.value/rhs)>(
										lhs.value / rhs)
						))

template<int L0, int L1, int L2, int L3, int L4, int L5, int L6, typename TL,
		typename TR>
inline auto operator/(TL const &lhs,
		DimensionedQuantity<L0, L1, L2, L3, L4, L5, L6, TR> const & rhs,
		)
				DECL_RET_TYPE(
						(
								DimensionedQuantity<-L0, -L1, -L2, -L3, -L4, -L5, -L6, decltype(lhs/rhs.value)>(
										lhs / rhs.value)
						))

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
	typedef DimensionedQuantity<0, 0, 0, 0, 0, 0, 0, double> DIMESIONLESS;
	typedef DimensionedQuantity<1, 0, 0, 0, 0, 0, 0, double> Length;
	typedef DimensionedQuantity<0, 1, 0, 0, 0, 0, 0, double> Mass;
	typedef DimensionedQuantity<0, 0, 1, 0, 0, 0, 0, double> Time;
	typedef DimensionedQuantity<0, 0, 0, 1, 0, 0, 0, double> Charge;
	typedef DimensionedQuantity<0, 0, 0, 0, 1, 0, 0, double> Temperature;
	typedef DimensionedQuantity<0, 0, 0, 0, 0, 1, 0, double> Luminousintensity;
	typedef DimensionedQuantity<0, 0, 0, 0, 0, 0, 1, double> AmountOfSubstance;
	typedef DimensionedQuantity<0, 2, 0, 0, 0, 0, 0, double> Area;
	typedef DimensionedQuantity<-2, -1, 2, 2, 0, 0, 0, double> Capacitance;
	typedef DimensionedQuantity<-3, 0, 0, 1, 0, 0, 0, double> ChargeDensity;
	typedef DimensionedQuantity<-2, -1, 1, 2, 0, 0, 0, double> ElectricConductance;
	typedef DimensionedQuantity<-3, -1, 1, 2, 0, 0, 0, double> Conductivity;
	typedef DimensionedQuantity<0, 0, -1, 1, 0, 0, 0, double> CurrentDensity;
	typedef DimensionedQuantity<-3, 0, 0, 0, 0, 0, 0, double> Density;
	typedef DimensionedQuantity<-2, 0, 0, 1, 0, 0, 0, double> Displacement;
	typedef DimensionedQuantity<1, 1, -2, -1, 0, 0, 0, double> ElectricField;
	typedef DimensionedQuantity<2, 1, -2, -1, 0, 0, 0, double> Electromotance;
	typedef DimensionedQuantity<2, 1, -2, 0, 0, 0, 0, double> Energy;
	typedef DimensionedQuantity<-1, 1, -2, 0, 0, 0, 0, double> EnergyDensity;
	typedef DimensionedQuantity<1, 1, -2, 0, 0, 0, 0, double> Force;
	typedef DimensionedQuantity<0, 0, -1, 0, 0, 0, 0, double> Frequency;
	typedef DimensionedQuantity<2, 1, -1, -2, 0, 0, 0, double> Impedance;
	typedef DimensionedQuantity<2, 1, 0, -2, 0, 0, 0, double> Inductance;
	typedef DimensionedQuantity<-1, 0, -1, 1, 0, 0, 0, double> MagneticIntensity;
	typedef DimensionedQuantity<2, 1, -1, -1, 0, 0, 0, double> MagneticFlux;
	typedef DimensionedQuantity<0, 1, -1, -1, 0, 0, 0, double> MagneticInduction;
	typedef DimensionedQuantity<2, 0, -1, 1, 0, 0, 0, double> MagneticMoment;
	typedef DimensionedQuantity<-1, 0, -1, 1, 0, 0, 0, double> Magnetization;
	typedef DimensionedQuantity<0, 0, -1, 1, 0, 0, 0, double> Magnetomotance;
	typedef DimensionedQuantity<1, 1, -1, 0, 0, 0, 0, double> Momentum;
	typedef DimensionedQuantity<-2, 1, -1, 0, 0, 0, 0, double> MomentumDensity;
	typedef DimensionedQuantity<1, 1, 0, -2, 0, 0, 0, double> Permeability;
	typedef DimensionedQuantity<-3, -1, 2, 2, 0, 0, 0, double> Permittivity;
	typedef DimensionedQuantity<-2, 0, 0, 1, 0, 0, 0, double> Polarization;
	typedef DimensionedQuantity<2, 1, -2, -1, 0, 0, 0, double> Potential;
	typedef DimensionedQuantity<2, 1, -3, 0, 0, 0, 0, double> Power;
	typedef DimensionedQuantity<-1, 1, -3, 0, 0, 0, 0, double> PowerDensity;
	typedef DimensionedQuantity<-1, 1, -2, 0, 0, 0, 0, double> Pressure;
	typedef DimensionedQuantity<-2, -1, 0, 2, 0, 0, 0, double> Reluctance;
	typedef DimensionedQuantity<2, 1, -1, -2, 0, 0, 0, double> Resistance;
	typedef DimensionedQuantity<3, 1, -1, -2, 0, 0, 0, double> Resistivity;
	typedef DimensionedQuantity<1, 1, -3, 0, 0, 0, 0, double> ThermalConductivity;
	typedef DimensionedQuantity<1, 1, -1, -1, 0, 0, 0, double> VectorPotential;
	typedef DimensionedQuantity<-1, 1, -1, 0, 0, 0, 0, double> Viscosity;
	typedef DimensionedQuantity<3, 0, 0, 0, 0, 0, 0, double> Volume;
	typedef DimensionedQuantity<0, 0, -1, 0, 0, 0, 0, double> Vorticity;
	typedef DimensionedQuantity<2, 1, -2, 0, 0, 0, 0, double> Work;

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
} //namespace simpla

#endif /* DIMENSIONED_QUANTITY_H_ */
