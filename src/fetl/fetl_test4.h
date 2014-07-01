/*
 * fetl_test4.h
 *
 *  Created on: 2014年3月24日
 *      Author: salmon
 */

#ifndef FETL_TEST4_H_
#define FETL_TEST4_H_

#include <gtest/gtest.h>
#include <random>
#include "fetl_test.h"
#include "fetl.h"

using namespace simpla;

TEST_P(TestFETL ,vec_zero_form)
{

	nTuple<3, scalar_type> vc1 = { 1.0, 2.0, 3.0 };

	nTuple<3, scalar_type> vc2 = { -1.0, 4.0, 2.0 };

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	auto res_scalar_field = mesh.make_field<VERTEX, scalar_type>();

	auto vaf = mesh.make_field<VERTEX, nTuple<3, scalar_type> >();
	auto vbf = mesh.make_field<VERTEX, nTuple<3, scalar_type> >();

	auto res_vector_field = mesh.make_field<VERTEX, nTuple<3, scalar_type> >();

	vaf.initialize();
	vbf.initialize();

	for (auto & p : vaf)
	{
		p = vc1 * uniform_dist(gen);
	}
	for (auto & p : vbf)
	{
		p = vc2 * uniform_dist(gen);
	}

	LOG_CMD(res_vector_field = Cross(vaf, vbf));

	for (auto s : mesh.Select(VERTEX))
	{
		ASSERT_EQ(Cross(vaf[s], vbf[s]), res_vector_field[s]);

	}

	LOG_CMD(res_scalar_field = Dot(vaf, vbf));

	for (auto s : mesh.Select(VERTEX))
	{
		ASSERT_EQ(InnerProductNTuple(vaf[s], vbf[s]), res_scalar_field[s]);
	}

}
#endif /* FETL_TEST4_H_ */
