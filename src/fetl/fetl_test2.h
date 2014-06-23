/*
 * fetl_test2.h
 *
 *  Created on: 2014年3月24日
 *      Author: salmon
 */

#ifndef FETL_TEST2_H_
#define FETL_TEST2_H_

#include <random>
#include <gtest/gtest.h>

#include "../utilities/log.h"
#include "../utilities/primitives.h"
#include "fetl.h"
#include "fetl_test_suit.h"

using namespace simpla;

TEST_P(TestFETL, vector_arithmetic)
{
	Field<mesh_type, VERTEX, value_type> f0(mesh);
	Field<mesh_type, EDGE, value_type> f1a(mesh), f1b(mesh);
	Field<mesh_type, FACE, value_type> f2a(mesh), f2b(mesh);
	Field<mesh_type, VOLUME, value_type> f3(mesh);

	Real ra = 1.0, rb = 10.0, rc = 100.0;
	value_type va, vb, vc;

	va = ra;
	vb = rb;
	vc = rc;

	f0.Init();
	f1a.Init();

	f1b.Init();
	f2a.Init();
	f2b.Init();
	f3.Init();

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
