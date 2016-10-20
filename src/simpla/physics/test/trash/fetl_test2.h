/*
 * fetl_test2.h
 *
 *  created on: 2014-3-24
 *      Author: salmon
 */

#ifndef FETL_TEST2_H_
#define FETL_TEST2_H_

#include <random>
#include <gtest/gtest.h>
#include "../../diff_geometry/calculus.h"
#include "field.h"
#include "../utilities/log.h"
#include "../utilities/primitives.h"

using namespace simpla;

template<typename TField>
class TestField: public testing::Test
{
protected:
	virtual void SetUp()
	{
		LOGGER.set_stdout_visable_level(10);

		domain_type(12, 20).swap(domain);

	}
public:

	typedef typename TField::domain_type domain_type;
	typedef typename TField::value_type value_type;

	domain_type domain;
	value_type default_value;

	typedef Field<domain_type, value_type> field_type;

	Field<domain_type, value_type> make_field() const
	{
		return std::move(Field<domain_type, value_type>(domain));
	}

	Field<domain_type, Real> make_scalarField() const
	{
		return std::move(Field<domain_type, Real>(domain));
	}

};

TYPED_TEST_CASE_P(TestField);

TYPED_TEST_P(TestField, vector_arithmetic)
{
	if (!mesh.is_valid())
		return;

	auto f0 = mesh.make_field<VERTEX, value_type>();
	auto f1 = mesh.make_field<EDGE, value_type>();
	auto f1a = mesh.make_field<EDGE, value_type>();
	auto f1b = mesh.make_field<EDGE, value_type>();
	auto f2a = mesh.make_field<FACE, value_type>();
	auto f2b = mesh.make_field<FACE, value_type>();
	auto f3 = mesh.make_field<VOLUME, value_type>();

	Real ra = 1.0, rb = 10.0, rc = 100.0;
	value_type va, vb, vc;

	va = ra;
	vb = rb;
	vc = rc;

	f0.clear();
	f1a.clear();

	f1b.clear();
	f2a.clear();
	f2b.clear();
	f3.clear();

	size_t count = 0;

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	for (auto & v : f1a)
	{
		v = va * uniform_dist(gen);
	}
	for (auto & v : f2a)
	{
		v = vb * uniform_dist(gen);
	}

	for (auto & v : f3)
	{
		v = vc * uniform_dist(gen);
	}

	LOG_CMD(f2b = cross(f1a, f1b));
	LOG_CMD(f3 = dot(f1a, f2a));
	LOG_CMD(f3 = dot(f2a, f1a));
	LOG_CMD(f3 = inner_product(f2a, f2a));
	LOG_CMD(f3 = inner_product(f1a, f1a));

	LOG_CMD(f0 = wedge(f0, f0));
	LOG_CMD(f1b = wedge(f0, f1a));
	LOG_CMD(f1b = wedge(f1a, f0));
	LOG_CMD(f2b = wedge(f0, f2a));
	LOG_CMD(f2b = wedge(f2a, f0));
	LOG_CMD(f3 = wedge(f0, f3));
	LOG_CMD(f3 = wedge(f3, f0));

	LOG_CMD(f2a = wedge(f1a, f1b));
	LOG_CMD(f3 = wedge(f1a, f2b));
	LOG_CMD(f3 = wedge(f2a, f1b));

}

#endif /* FETL_TEST2_H_ */
