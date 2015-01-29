/*
 * field_test.h
 *
 *  Created on: 2015年1月29日
 *      Author: salmon
 */

#ifndef CORE_FIELD_TEST_FIELD_TEST_H_
#define CORE_FIELD_TEST_FIELD_TEST_H_

#include <gtest/gtest.h>
#include <tuple>

#include "../../utilities/utilities.h"
#include "../field.h"
#include "../../diff_geometry/simple_mesh.h"
using namespace simpla;

template<typename TField>
class TestField: public testing::TestWithParam<typename TField::mesh_type>
{

protected:
	void SetUp()
	{
		LOGGER.set_stdout_visable_level(LOG_VERBOSE);
	}

	void TearDown()
	{
	}
public:

	typedef TField field_type;
	typedef typename TField::mesh_type mesh_type;
	typedef typename TField::value_type value_type;

	typedef Field<mesh_type, value_type> scalar_field_type;

	mesh_type mesh;

	field_type make_field() const
	{
		return std::move(field_type(mesh));
	}
	scalar_field_type make_scalar_field() const
	{
		return std::move(scalar_field_type(mesh));
	}

};

typedef TestField<_Field<SimpleMesh, std::shared_ptr<double>>> TestFieldCase;

TEST_P(TestFieldCase, assign)
{

	auto f1 = make_field();

	value_type va;

	va = 2.0;

	f1 = va;

	size_t count = 0;
	for (auto const &s : mesh)
	{
		++count;
		EXPECT_LE(mod(va - f1[s]), EPSILON);
	}
	EXPECT_EQ(count, mesh.max_hash());
}

//TEST_P(TestFieldCase, index)
//{
//
//	auto f1 = make_field();
//
//	f1.clear();
//
//	value_type va;
//
//	va = 2.0;
//
//	for (auto const &s : mesh)
//	{
//		f1[s] = va * mesh.hash(s);
//	}
//
//	for (auto s : mesh)
//	{
//		EXPECT_LE(mod(va * mesh.hash(s) - f1[s]), EPSILON);
//	}
//}
//
//TEST_P(TestFieldCase, constant_real)
//{
//
//	auto f1 = make_field();
//	auto f2 = make_field();
//	auto f3 = make_field();
//
//	f3 = 1;
//	Real a, b, c;
//	a = 1.0, b = -2.0, c = 3.0;
//
//	value_type va, vb;
//
//	va = 2.0;
//	vb = 3.0;
//
//	f1 = va;
//	f2 = vb;
//
//	LOG_CMD(f3 = -f1 *a +f2*c - f1/b -f1/**/);
//
//	for (auto const& s : mesh)
//	{
//		value_type res;
//		res = -f1[s] * a + f2[s] * c - f1[s] / b - f1[s];
//
//		EXPECT_LE(mod( res- f3[s]),EPSILON) << res << " " << f1[s];
//	}
//}
//
//TEST_P(TestFieldCase, scalar_field)
//{
//
//	auto f1 = make_field();
//	auto f2 = make_field();
//	auto f3 = make_field();
//	auto f4 = make_field();
//
//	auto a = make_scalar_field();
//	auto b = make_scalar_field();
//	auto c = make_scalar_field();
//
//	Real ra = 1.0, rb = 10.0, rc = 100.0;
//
//	value_type va, vb, vc;
//
//	va = ra;
//	vb = rb;
//	vc = rc;
//
//	a = ra;
//	b = rb;
//	c = rc;
//
//	f1.clear();
//	f2.clear();
//	f3.clear();
//	f4.clear();
//
//	size_t count = 0;
//
//	std::mt19937 gen;
//	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);
//
//	for (auto s : f1.mesh())
//	{
//		f1[s] = va * uniform_dist(gen);
//	}
//	for (auto s : f2.mesh())
//	{
//		f2[s] = vb * uniform_dist(gen);
//	}
//
//	for (auto s : f3.mesh())
//	{
//		f3[s] = vc * uniform_dist(gen);
//	}
//
//	LOG_CMD(f4 = -f1 * a + f2 * b - f3 / c - f1);
//
////	Plus( Minus(Negate(Wedge(f1,a)),Divides(f2,b)),Multiplies(f3,c) )
//
//	/**           (+)
//	 *           /   \
//		 *         (-)    (*)
//	 *        /   \    | \
//		 *      (^)    (/) f1 c
//	 *     /  \   /  \
//		 *   -f1      a f2   b
//	 *
//	 * */
//	count = 0;
//
//	for (auto s : mesh)
//	{
//		value_type res = -f1[s] * ra + f2[s] * rb - f3[s] / rc - f1[s];
//
//		EXPECT_LE( mod(res-f4[s]) ,EPSILON ) << "s= " << (mesh.hash(s));
//	}
//
//	EXPECT_EQ(0,count) << "number of error points =" << count;
//}
#endif /* CORE_FIELD_TEST_FIELD_TEST_H_ */
