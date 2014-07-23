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

#include "../utilities/log.h"
#include "../utilities/primitives.h"
#include "fetl.h"
#include "fetl_test.h"

using namespace simpla;

TEST_P(TestFETL, vector_arithmetic)
{
	if (!mesh.is_valid()) return;

	auto f0 = mesh.template make_field<VERTEX, value_type>();
	auto f1 = mesh.template make_field<EDGE, value_type>();
	auto f1a = mesh.template make_field<EDGE, value_type>();
	auto f1b = mesh.template make_field<EDGE, value_type>();
	auto f2a = mesh.template make_field<FACE, value_type>();
	auto f2b = mesh.template make_field<FACE, value_type>();
	auto f3 = mesh.template make_field<VOLUME, value_type>();

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

	update_ghosts(&f1a);
	update_ghosts(&f2a);
	update_ghosts(&f3);

	LOG_CMD(f2b = Cross(f1a, f1b));
	LOG_CMD(f3 = Dot(f1a, f2a));
	LOG_CMD(f3 = Dot(f2a, f1a));
	LOG_CMD(f3 = InnerProduct(f2a, f2a));
	LOG_CMD(f3 = InnerProduct(f1a, f1a));

	LOG_CMD(f0 = Wedge(f0, f0));
	LOG_CMD(f1b = Wedge(f0, f1a));
	LOG_CMD(f1b = Wedge(f1a, f0));
	LOG_CMD(f2b = Wedge(f0, f2a));
	LOG_CMD(f2b = Wedge(f2a, f0));
	LOG_CMD(f3 = Wedge(f0, f3));
	LOG_CMD(f3 = Wedge(f3, f0));

	LOG_CMD(f2a = Wedge(f1a, f1b));
	LOG_CMD(f3 = Wedge(f1a, f2b));
	LOG_CMD(f3 = Wedge(f2a, f1b));

}

#endif /* FETL_TEST2_H_ */
