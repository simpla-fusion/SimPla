/*
 * fetl_test4.h
 *
 *  created on: 2014-3-24
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

	if (!manifold.is_valid()) return;

	nTuple<3, scalar_type> vc1 = { 1.0, 2.0, 3.0 };

	nTuple<3, scalar_type> vc2 = { -1.0, 4.0, 2.0 };

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	auto res_scalar_field = manifold.make_field<VERTEX, scalar_type>();

	auto vaf = manifold.make_field<VERTEX, nTuple<3, scalar_type> >();
	auto vbf = manifold.make_field<VERTEX, nTuple<3, scalar_type> >();

	auto res_vector_field = manifold.make_field<VERTEX, nTuple<3, scalar_type> >();

	vaf.clear();
	vbf.clear();

	for (auto & p : vaf)
	{
		p = vc1 * uniform_dist(gen);
	}
	update_ghosts(&vaf);
	for (auto & p : vbf)
	{
		p = vc2 * uniform_dist(gen);
	}
	update_ghosts(&vbf);

	LOG_CMD(res_vector_field = Cross(vaf, vbf));

	for (auto s : manifold.select(VERTEX))
	{
		ASSERT_EQ(Cross(vaf[s], vbf[s]), res_vector_field[s]);

	}

	LOG_CMD(res_scalar_field = Dot(vaf, vbf));

	for (auto s : manifold.select(VERTEX))
	{
		ASSERT_EQ(InnerProductNTuple(vaf[s], vbf[s]), res_scalar_field[s]);
	}

}
#endif /* FETL_TEST4_H_ */
