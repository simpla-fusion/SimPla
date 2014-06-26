/*
 * fetl_test3.h
 *
 *  Created on: 2014年3月24日
 *      Author: salmon
 */

#ifndef FETL_TEST3_H_
#define FETL_TEST3_H_

#include <gtest/gtest.h>
#include "field_io_test.h"
#include "fetl_test.h"

using namespace simpla;

static constexpr auto epsilon = 1e7 * std::numeric_limits<Real>::epsilon();

TEST_P(TestFETL, grad0)
{

	Real error = 0.5 * std::pow(InnerProductNTuple(K, mesh.GetDx()), 2.0);

	Field<mesh_type, VERTEX, scalar_type> f0(mesh);

	Field<mesh_type, EDGE, scalar_type> f1(mesh);
	Field<mesh_type, EDGE, scalar_type> f1b(mesh);

	f0.Clear();
	f1.Clear();
	f1b.Clear();
	for (auto s : mesh.Select(VERTEX))
	{
		f0[s] = std::sin(InnerProductNTuple(K, mesh.CoordinatesToCartesian(mesh.GetCoordinates(s))));
	};
	LOG_CMD(f1 = Grad(f0));
	Real m = 0.0;
	Real variance = 0;
	scalar_type average;
	average *= 0.0;

	for (auto s : mesh.Select(EDGE))
	{

		auto expect = mesh.Sample(Int2Type<EDGE>(), s, K)
		        * std::cos(InnerProductNTuple(K, mesh.CoordinatesToCartesian(mesh.GetCoordinates(s))));

		f1b[s] = expect;

		variance += abs((f1[s] - expect) * (f1[s] - expect));

		average += (f1[s] - expect);

		if (abs(f1[s]) > epsilon || abs(expect) > epsilon)
		{
//			if (error < abs(2.0 * (f1[s] - expect) / (f1[s] + expect)))
//			{
//				CHECK(mesh.Sample(Int2Type<EDGE>(), s, K));
//				CHECK(InnerProductNTuple(K, mesh.CoordinatesToCartesian(mesh.GetCoordinates(s))));
//				CHECK(f1[s]);
//				CHECK(expect);
//				CHECK(error);
//			}
			ASSERT_GE(error, abs(2.0 * (f1[s] - expect) / (f1[s] + expect)));

		}

	}

	variance /= f1.size();
	average /= f1.size();
	CHECK(variance);
	CHECK(average);
}

TEST_P(TestFETL, grad3)
{

	Real error = 0.5 * std::pow(InnerProductNTuple(K, mesh.GetDx()), 2.0);

	Field<mesh_type, FACE, scalar_type> f2(mesh);
	Field<mesh_type, FACE, scalar_type> f2b(mesh);
	Field<mesh_type, VOLUME, scalar_type> f3(mesh);

	f3.Clear();
	f2.Clear();
	f2b.Clear();

	for (auto s : mesh.Select(VOLUME))
	{
		f3[s] = std::sin(InnerProductNTuple(K, mesh.GetCoordinates(s)));
	};

	LOG_CMD(f2 = Grad(f3));

	Real m = 0.0;
	Real variance = 0;
	scalar_type average;
	average *= 0.0;

	for (auto s : mesh.Select(FACE))
	{

		auto expect = std::cos(InnerProductNTuple(K, mesh.GetCoordinates(s))) * K[mesh.ComponentNum(s)];

		f2b[s] = expect;

		variance += abs((f2[s] - expect) * (f2[s] - expect));

		average += (f2[s] - expect);

		if (abs(f2[s]) > epsilon || abs(expect) > epsilon)
		{
			if (abs(2.0 * (f2[s] - expect) / (f2[s] + expect)) > error)
			{
				CHECK(f2[s]);
				CHECK(expect);
				CHECK(mesh.GetCoordinates(s));
			}
			EXPECT_LE(abs(2.0 * (f2[s] - expect) / (f2[s] + expect)), error);
		}

	}

	variance /= f2.size();
	average /= f2.size();
	CHECK(variance);
	CHECK(average);

}

TEST_P(TestFETL, diverge1)
{

	auto error = 0.5 * std::pow(InnerProductNTuple(K, mesh.GetDx()), 2.0);

	Field<mesh_type, EDGE, scalar_type> f1(mesh);
	Field<mesh_type, VERTEX, scalar_type> f0(mesh);

	f0.Clear();
	f1.Clear();

	for (auto s : mesh.Select(EDGE))
	{
		f1[s] = std::sin(InnerProductNTuple(K, mesh.GetCoordinates(s)));
	};
	f0 = Diverge(f1);

	Real variance = 0;

	scalar_type average = 0.0;

	for (auto s : mesh.Select(VERTEX))
	{

		auto expect = std::cos(InnerProductNTuple(K, mesh.GetCoordinates(s))) * (K[0] + K[1] + K[2]);

		variance += abs((f0[s] - expect) * (f0[s] - expect));

		average += (f0[s] - expect);

		auto x = mesh.GetCoordinates(s);

		if ((abs(f0[s]) > epsilon || abs(expect) > epsilon))
		{
//			if (abs(2.0 * (f0[s] - expect) / (f0[s] + expect)) > error)
//			{
//				CHECK(f0[s]);
//				CHECK(expect);
//				CHECK(mesh.GetCoordinates(s));
//			}
			EXPECT_LE(abs(2.0 * (f0[s] - expect) / (f0[s] + expect)), error);
		}

	}

	variance /= f0.size();
	average /= f0.size();
	CHECK(variance);
	CHECK(average);

}

TEST_P(TestFETL, diverge2)
{

	auto error = 0.5 * std::pow(InnerProductNTuple(K, mesh.GetDx()), 2.0);

	Field<mesh_type, FACE, scalar_type> f2(mesh);
	Field<mesh_type, VOLUME, scalar_type> f3(mesh);

	f3.Clear();
	f2.Clear();

	for (auto s : mesh.Select(FACE))
	{
		f2[s] = std::sin(InnerProductNTuple(K, mesh.GetCoordinates(s)));
	};

	f3 = Diverge(f2);

	Real variance = 0;
	scalar_type average = 0.0;

	for (auto s : mesh.Select(VOLUME))
	{

		auto expect = std::cos(InnerProductNTuple(K, mesh.GetCoordinates(s))) * (K[0] + K[1] + K[2]);

		variance += abs((f3[s] - expect) * (f3[s] - expect));

		average += (f3[s] - expect);

		if (abs(f3[s]) > epsilon || abs(expect) > epsilon)
			ASSERT_LE(abs(2.0 * (f3[s] - expect) / (f3[s] + expect)), error);

	}

	variance /= f3.size();
	average /= f3.size();
	CHECK(variance);
	CHECK(average);

}

TEST_P(TestFETL, curl1)
{

	auto error = std::pow(InnerProductNTuple(K, mesh.GetDx()), 2.0);

	Field<mesh_type, EDGE, scalar_type> vf1(mesh);
	Field<mesh_type, EDGE, scalar_type> vf1b(mesh);
	Field<mesh_type, FACE, scalar_type> vf2(mesh);
	Field<mesh_type, FACE, scalar_type> vf2b(mesh);

	vf1.Clear();
	vf1b.Clear();
	vf2.Clear();
	vf2b.Clear();

	Real m = 0.0;
	Real variance = 0;
	scalar_type average;
	average *= 0.0;

	for (auto s : mesh.Select(EDGE))
	{
		vf1[s] = std::sin(InnerProductNTuple(K, mesh.GetCoordinates(s)));
	};
//	GLOBAL_DATA_STREAM.OpenFile("FetlTest");
//	GLOBAL_DATA_STREAM.OpenGroup("/curl1");
//	LOGGER << SAVE(vf1);
	LOG_CMD(vf2 = Curl(vf1));
//	LOGGER << SAVE(vf2);
	for (auto s : mesh.Select(FACE))
	{
		auto n = mesh.ComponentNum(s);

		auto expect = std::cos(InnerProductNTuple(K, mesh.GetCoordinates(s))) * (K[(n + 1) % 3] - K[(n + 2) % 3]);

		variance += abs((vf2[s] - expect) * (vf2[s] - expect));

		average += (vf2[s] - expect);
		auto x = mesh.GetCoordinates(s);

		if ((abs(vf2[s]) > epsilon || abs(expect) > epsilon))
		{
//			if (abs(2.0 * (vf2[s] - expect) / (vf2[s] + expect)) > error)
//			{
//				CHECK(n);
//				CHECK(K);
//				CHECK(mesh.GetCoordinates(s));
//				CHECK(std::cos(InnerProductNTuple(K, mesh.GetCoordinates(s))));
//				CHECK(vf2[s]);
//				CHECK(expect);
//			}
			ASSERT_LE(abs(2.0 * (vf2[s] - expect) / (vf2[s] + expect)), error);

		}

	}

	variance /= vf2.size();
	average /= vf2.size();
	CHECK(variance);
	CHECK(average);

}

TEST_P(TestFETL, curl2)
{
	auto error = std::pow(InnerProductNTuple(K, mesh.GetDx()), 2.0);

	Field<mesh_type, EDGE, scalar_type> vf1(mesh);
	Field<mesh_type, EDGE, scalar_type> vf1b(mesh);
	Field<mesh_type, FACE, scalar_type> vf2(mesh);
	Field<mesh_type, FACE, scalar_type> vf2b(mesh);

	vf1.Clear();
	vf1b.Clear();
	vf2.Clear();
	vf2b.Clear();

	Real m = 0.0;
	Real variance = 0;
	scalar_type average;
	average *= 0.0;

	for (auto s : mesh.Select(FACE))
	{
		vf2[s] = std::sin(InnerProductNTuple(K, mesh.GetCoordinates(s)));
	};

	LOG_CMD(vf1 = Curl(vf2));

	vf1b.Clear();

	for (auto s : mesh.Select(EDGE))
	{

		auto n = mesh.ComponentNum(s);

		auto expect = std::cos(InnerProductNTuple(K, mesh.GetCoordinates(s))) * (K[(n + 1) % 3] - K[(n + 2) % 3]);

		vf1b[s] = expect;

		variance += abs((vf1[s] - expect) * (vf1[s] - expect));

		average += (vf1[s] - expect);

		auto x = mesh.GetCoordinates(s);
		if ((abs(vf1[s]) > epsilon || abs(expect) > epsilon))
			ASSERT_LE(abs(2.0 * (vf1[s] - expect) / (vf1[s] + expect)), error);

	}

	variance /= vf1.size();
	average /= vf1.size();
	CHECK(variance);
	CHECK(average);

}

TEST_P(TestFETL, identity_curl_grad_f0_eq_0)
{

	Field<mesh_type, VERTEX, scalar_type> f0(mesh);

	Field<mesh_type, EDGE, scalar_type> f1(mesh);
	Field<mesh_type, FACE, scalar_type> f2a(mesh);
	Field<mesh_type, FACE, scalar_type> f2b(mesh);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m = 0.0;
	f0.Clear();
	for (auto s : mesh.Select(VERTEX))
	{

		auto a = uniform_dist(gen);
		f0[s] = default_value * a;
		m += a * a;
	}

	m = std::sqrt(m) * abs(default_value);

	LOG_CMD(f1 = Grad(f0));
	LOG_CMD(f2a = Curl(f1));
	LOG_CMD(f2b = Curl(Grad(f0)));

	size_t count = 0;
	Real relative_error = 0;

	for (auto s : mesh.Select(FACE))
	{

		relative_error += abs(f2b[s]);
		ASSERT_EQ((f2a[s]), (f2b[s]));
	}

	relative_error /= m;

	INFORM2(relative_error);
	ASSERT_LE(relative_error, epsilon)<<m;

}

TEST_P(TestFETL, identity_curl_grad_f3_eq_0)
{

	Field<mesh_type, VOLUME, scalar_type> f3(mesh);
	Field<mesh_type, EDGE, scalar_type> f1a(mesh);
	Field<mesh_type, EDGE, scalar_type> f1b(mesh);
	Field<mesh_type, FACE, scalar_type> f2(mesh);
	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m = 0.0;

	f3.Clear();

	for (auto s : mesh.Select(VOLUME))
	{
		auto a = uniform_dist(gen);
		f3[s] = a * default_value;
		m += a * a;
	}

	m = std::sqrt(m) * abs(default_value);

	LOG_CMD(f2 = Grad(f3));
	LOG_CMD(f1a = Curl(f2));
	LOG_CMD(f1b = Curl(Grad(f3)));

	size_t count = 0;
	Real relative_error = 0;

	for (auto s : mesh.Select(EDGE))
	{

		ASSERT_EQ((f1a[s]), (f1b[s]));

		relative_error += abs(f1b[s]);

	}

	relative_error /= m;

	INFORM2(relative_error);
	ASSERT_LE(relative_error, epsilon);

}

TEST_P(TestFETL, identity_div_curl_f1_eq0)
{

	Field<mesh_type, EDGE, scalar_type> f1(mesh);
	Field<mesh_type, FACE, scalar_type> f2(mesh);
	Field<mesh_type, VERTEX, scalar_type> f0a(mesh);
	Field<mesh_type, VERTEX, scalar_type> f0b(mesh);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	f2.Clear();

	Real m = 0.0;

	for (auto s : mesh.Select(FACE))
	{
		auto a = uniform_dist(gen);

		f2[s] = default_value * uniform_dist(gen);

		m += a * a;
	}

	m = std::sqrt(m) * abs(default_value);

	LOG_CMD(f1 = Curl(f2));

	LOG_CMD(f0a = Diverge(f1));

	LOG_CMD(f0b = Diverge(Curl(f2)));

	size_t count = 0;
	Real relative_error = 0;

	for (auto s : mesh.Select(VERTEX))
	{
		relative_error += abs(f0b[s]);
		ASSERT_EQ((f0a[s]), (f0b[s]));
	}

	relative_error /= m;
	INFORM2(relative_error);
	ASSERT_LE(relative_error, epsilon);

}

TEST_P(TestFETL, identity_div_curl_f2_eq0)
{

	Field<mesh_type, EDGE, scalar_type> f1(mesh);
	Field<mesh_type, FACE, scalar_type> f2(mesh);
	Field<mesh_type, VOLUME, scalar_type> f3a(mesh);
	Field<mesh_type, VOLUME, scalar_type> f3b(mesh);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	f1.Clear();

	Real m = 0.0;

	for (auto s : mesh.Select(EDGE))
	{
		auto a = uniform_dist(gen);
		f1[s] = default_value * a;
		m += a * a;
	}

	m = std::sqrt(m) * abs(default_value);

	LOG_CMD(f2 = Curl(f1));

	LOG_CMD(f3a = Diverge(f2));

	LOG_CMD(f3b = Diverge(Curl(f1)));

	size_t count = 0;

	Real relative_error = 0;

	for (auto s : mesh.Select(VOLUME))
	{

		ASSERT_DOUBLE_EQ(abs(f3a[s]), abs(f3b[s]));

		relative_error += abs(f3b[s]);

	}

	relative_error /= m;
	INFORM2(relative_error);
	ASSERT_LE(relative_error, epsilon);

}

#endif /* FETL_TEST3_H_ */
