/*
 * units_test.cpp
 *
 *  Created on: 2012-3-3
 *      Author: salmon
 */

#include "physics.h"
#include <iostream>
#include <gtest/gtest.h>

using namespace physics;
class TestPhysicsUnitSystem: public testing::Test
{

protected:
	virtual void SetUp()
	{

	}

};
#define DECLARE_TEST_OF_UNIT(_L,_R)                            \
	EXPECT_TRUE( unit_sys::ToSIUnit(unit_sys::_L)== _R)        \
	<< unit_sys::ToSIUnit(unit_sys::_L)<<"~="<<_R;             \
	EXPECT_TRUE( unit_sys::_L==unit_sys::FromSIUnit(_R))       \
	 <<unit_sys::_L<<"~="<< unit_sys::FromSIUnit(_R);

#define DECLARE_TEST_UNITS                                                                                  \
	DECLARE_TEST_OF_UNIT( m, m);                                                                   \
                                                                                                            \
	DECLARE_TEST_OF_UNIT( g, g);                                                                   \
	DECLARE_TEST_OF_UNIT( s, s);                                                                   \
	DECLARE_TEST_OF_UNIT( C, C);                                                                   \
	DECLARE_TEST_OF_UNIT( A, A);                                                                   \
	DECLARE_TEST_OF_UNIT( K, K);                                                                   \
	DECLARE_TEST_OF_UNIT( cd, cd);                                                                 \
	DECLARE_TEST_OF_UNIT( mol, mol);                                                               \
	DECLARE_TEST_OF_UNIT( F, F);                                                                   \
                                                                                                            \
	DECLARE_TEST_OF_UNIT( speed_of_light, SI_speed_of_light * m / s);                              \
	DECLARE_TEST_OF_UNIT( permeability_of_free_space,                                              \
			SI_permeability_of_free_space * H / m);                                                         \
	DECLARE_TEST_OF_UNIT( permittivity_of_free_space,                                              \
			1.0/(speed_of_light*speed_of_light*permeability_of_free_space));                                \
	DECLARE_TEST_OF_UNIT( gravitational_constant,                                                  \
			SI_gravitational_constant * (m * m * m) / (s * s)/ kg);                                         \
	DECLARE_TEST_OF_UNIT( plank_constant, SI_plank_constant * J * s);                              \
	DECLARE_TEST_OF_UNIT( plank_constant_bar,                                                      \
			SI_plank_constant_bar * J * s);                                                                 \
	DECLARE_TEST_OF_UNIT( elementary_charge, SI_elementary_charge * C);                            \
	DECLARE_TEST_OF_UNIT( electron_mass, SI_electron_mass * kg);                                   \
	DECLARE_TEST_OF_UNIT( proton_mass, SI_proton_mass* kg);                                        \
	DECLARE_TEST_OF_UNIT( electron_charge_mass_ratio,                                              \
			SI_electron_charge_mass_ratio * C / kg);                                                        \
	DECLARE_TEST_OF_UNIT( Rydberg_constant, SI_Rydberg_constant/m);                                \
	DECLARE_TEST_OF_UNIT( Avogadro_constant, SI_Avogadro_constant/mol);                            \
	DECLARE_TEST_OF_UNIT( Faraday_constant, SI_Faraday_constant*C/mol);                            \
	DECLARE_TEST_OF_UNIT( Boltzmann_constant,                                                      \
			SI_Boltzmann_constant * J / K);                                                                 \
	DECLARE_TEST_OF_UNIT( electron_volt, SI_electron_volt*J);                                      \
	DECLARE_TEST_OF_UNIT( atomic_mass_unit, atomic_mass_unit);                                     \
	EXPECT_EQ(permeability_of_free_space*permittivity_of_free_space,                                        \
			1.0/speed_of_light/speed_of_light)                                                              \
		;                                                                                                   \


TEST_F(TestPhysicsUnitSystem, arithmetic)
{
	using namespace physics::units::si;
	Velocity v;
	Length l = 1.0 * km;
	Time t = 2.0 * s;
	v = l / t;
	EXPECT_EQ(500.0 * m / s, v);
	EXPECT_DOUBLE_EQ(500.0, v / (m / s));
	EXPECT_DOUBLE_EQ(500.0 / SI_speed_of_light, v / (speed_of_light));
}

TEST_F(TestPhysicsUnitSystem, si)
{

	using namespace physics::units::si;
	namespace unit_sys = physics::units::si;

	EXPECT_DOUBLE_EQ(unit_sys::m.value(), 1.0);
	EXPECT_DOUBLE_EQ(unit_sys::kg.value(), 1.0);
	EXPECT_DOUBLE_EQ(unit_sys::s.value(), 1.0);

	DECLARE_TEST_UNITS
}
TEST_F(TestPhysicsUnitSystem, cgs)
{
	using namespace physics::units::si;
	namespace unit_sys = physics::units::cgs;

	EXPECT_DOUBLE_EQ(unit_sys::cm.value(), 1.0);
	EXPECT_DOUBLE_EQ(unit_sys::g.value(), 1.0);
	EXPECT_DOUBLE_EQ(unit_sys::s.value(), 1.0);

	DECLARE_TEST_UNITS
}
TEST_F(TestPhysicsUnitSystem, nature)
{
	using namespace physics::units::si;
	namespace unit_sys = physics::units::nature;

	EXPECT_DOUBLE_EQ(unit_sys::speed_of_light.value(), 1.0);
	EXPECT_DOUBLE_EQ(unit_sys::elementary_charge.value(), 1.0);
	EXPECT_DOUBLE_EQ(unit_sys::elementary_charge.value(), 1.0);
	EXPECT_DOUBLE_EQ(unit_sys::permittivity_of_free_space.value(), 1.0);
	EXPECT_DOUBLE_EQ(unit_sys::permeability_of_free_space.value(), 1.0);
	DECLARE_TEST_OF_UNIT(m, m)

	DECLARE_TEST_UNITS
}

//
//#include "fetl/fetl.h"
//using namespace simpla;
//template<typename TF>
//class TestFETLBasicArithmetic: public testing::Test
//{
//protected:
//	virtual void SetUp()
//	{
//		IVec3 dims =
//		{ 20, 30, 40 };
//
//		Vec3 xmin =
//		{ 0, 0, 0 };
//
//		Vec3 xmax =
//		{ 1, 1, 1 };
//
//		grid.Initialize(1.0, xmin, xmax, dims);
//
//	}
//public:
//	typedef TF FieldType;
//	typedef typename FieldType::Grid Grid;
//	typedef typename FieldType::ValueType ValueType;
//	typedef typename ComplexTraits<ValueType>::ValueType CValueType;
//	typedef Field<TF::IForm, CValueType, _FETL_Field<typename TF::Grid> > CFieldType;
//	typename FieldType::Grid grid;
//
//};
//
//typedef testing::Types<RZeroForm, ROneForm, RTwoForm //
//		, CZeroForm, COneForm, CTwoForm //
//		, VecZeroForm //, VecOneForm, VecTwoForm, VecThreeForm
//
//> AllFieldTypes;
//
//TYPED_TEST_CASE(TestFETLBasicArithmetic, AllFieldTypes);
//
//TYPED_TEST(TestFETLBasicArithmetic, dimensioned_field){
//{
//	using namespace physics;
//	using namespace physics::units;
//
//	typename TestFixture::Grid const & grid = TestFixture::grid;
//
//	typename TestFixture::FieldType f1( grid),f2(grid),f3(grid);
//
//	size_t num_of_comp=grid.get_num_of_comp(TestFixture::FieldType::IForm);
//
//	size_t size=grid.get_num_of_elements(TestFixture::FieldType::IForm);
//
//	ScalarField r(grid);
//
//	r=4.0;
//
//	auto a=1.0* si::cm/si::s;
//	auto b=2.0* si::km/si::ms;
//
//	f1 = 2.0;
//	f2 = 1.0;
//
//	typedef decltype(typename TestFixture::FieldType::ValueType()*(si::Velocity())) PhysicalValue;
//	Field<TestFixture::FieldType::IForm,PhysicalValue,_FETL_Field<typename TestFixture::Grid> > res(grid);
//
//	res=-f1*a-b*f2;
//	f3 = res/(si::m/si::s);
//
//	Real ta=a/(si::m/si::s);
//	Real tb=b/(si::m/si::s);
//
//	for (auto s = grid.get_center_elements_begin(TestFixture::FieldType::IForm);
//			s!=grid.get_center_elements_end(TestFixture::FieldType::IForm); ++s)
//	{
//		ASSERT_EQ((-f1[*s]*a- b*f2[*s]) , res[*s]) << "idx=" <<*s;
//		ASSERT_EQ(f3[*s] , res[*s].value()) << "idx=" <<*s;
//	}
//}
//
//}
