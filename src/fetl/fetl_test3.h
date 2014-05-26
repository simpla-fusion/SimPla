/*
 * fetl_test3.h
 *
 *  Created on: 2014年3月24日
 *      Author: salmon
 */

#ifndef FETL_TEST3_H_
#define FETL_TEST3_H_

#include <gtest/gtest.h>

#include <stddef.h>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <random>
#include <tuple>
#include <valarray>

#include "../mesh/mesh.h"
#include "../utilities/log.h"

#include "fetl_test.h"
#include "fetl.h"
#include "ntuple.h"
#include "primitives.h"

using namespace simpla;

static constexpr auto epsilon = 1e7 * std::numeric_limits<Real>::epsilon();

typedef Mesh<EuclideanGeometry<OcForest>> TMesh;

class TestDiffCalculus: public testing::TestWithParam<

std::tuple<

nTuple<TMesh::NDIMS, size_t>,

typename TMesh::coordinates_type,

typename TMesh::coordinates_type>

>
{

protected:
	virtual void SetUp()
	{
		auto param = GetParam();

		mesh.SetExtents(std::get<0>(param), std::get<1>(param), std::get<2>(param));

		SetDefaultValue(&default_value);
	}
public:

	typedef TMesh mesh_type;
	typedef Real value_type;
	typedef mesh_type::iterator iterator;
	typedef mesh_type::coordinates_type coordinates_type;
	typedef Field<mesh_type, VERTEX, value_type> TZeroForm;
	typedef Field<mesh_type, EDGE, value_type> TOneForm;
	typedef Field<mesh_type, FACE, value_type> TTwoForm;
	typedef Field<mesh_type, VOLUME, value_type> TThreeForm;

	mesh_type mesh;

	static constexpr double PI = 3.141592653589793;

	static constexpr nTuple<3, Real> K = { 2.0 * PI, 3.0 * PI, 4.0 * PI }; // @NOTE must   k = n TWOPI, period condition

	value_type default_value;

};

TEST_P(TestDiffCalculus, grad0)
{
	auto d = mesh.GetDimensions();
	nTuple<3, Real> k = K;

	for (int i = 0; i < mesh.NDIMS; ++i)
	{
		if (d[i] <= 1)
			k[i] = 0;
	}

	auto error = 0.5 * std::pow(Dot(k, mesh.GetDx()), 2.0);

	TOneForm f1(mesh);
	TOneForm f1b(mesh);
	TZeroForm f0(mesh);

	f0.Clear();
	f1.Clear();
	f1b.Clear();
	for (auto s : mesh.GetRange(VERTEX))
	{
		f0[s] = std::sin(Dot(k, mesh.GetCoordinates(s)));
	};

	LOG_CMD(f1 = Grad(f0));

	Real m = 0.0;
	Real variance = 0;
	value_type average;
	average *= 0.0;

	for (auto s : mesh.GetRange(EDGE))
	{

		auto expect = std::cos(Dot(k, mesh.GetCoordinates(s))) * k[mesh.ComponentNum(s.self_)];

		f1b[s] = expect;

		variance += abs((f1[s] - expect) * (f1[s] - expect));

		average += (f1[s] - expect);

		if (abs(f1[s]) > epsilon || abs(expect) > epsilon)
			ASSERT_LE( abs(2.0*(f1[s]-expect)/(f1[s] + expect)), error )<< " " << f1[s] << " " << expect << " "
			<< mesh.GetCoordinates(s);

		}

	variance /= f1.size();
	average /= f1.size();
	CHECK(variance);
	CHECK(average);
}

TEST_P(TestDiffCalculus, grad3)
{
	auto d = mesh.GetDimensions();

	nTuple<3, Real> k = K;

	for (int i = 0; i < mesh.NDIMS; ++i)
	{
		if (d[i] <= 1)
			k[i] = 0;
	}

	auto error = 0.5 * std::pow(Dot(k, mesh.GetDx()), 2.0);

	TTwoForm f2(mesh);
	TTwoForm f2b(mesh);
	TThreeForm f3(mesh);

	f3.Clear();
	f2.Clear();
	f2b.Clear();

	for (auto s : mesh.GetRange(VOLUME))
	{
		f3[s] = std::sin(Dot(k, mesh.GetCoordinates(s)));
	};

	LOG_CMD(f2 = Grad(f3));

	Real m = 0.0;
	Real variance = 0;
	value_type average;
	average *= 0.0;

	for (auto s : mesh.GetRange(FACE))
	{

		auto expect = std::cos(Dot(k, mesh.GetCoordinates(s))) * k[mesh.ComponentNum(s.self_)];

		f2b[s] = expect;

		variance += abs((f2[s] - expect) * (f2[s] - expect));

		average += (f2[s] - expect);

		if (abs(f2[s]) > epsilon || abs(expect) > epsilon)
			ASSERT_LE(abs(2.0*(f2[s]-expect)/(f2[s] + expect)), error )<< " " << f2[s] << " "
			<< f2b[s];

		}

	variance /= f2.size();
	average /= f2.size();
	CHECK(variance);
	CHECK(average);

}

TEST_P(TestDiffCalculus, diverge1)

{
	nTuple<3, Real> k = K;
	auto d = mesh.GetDimensions();

	for (int i = 0; i < mesh.NDIMS; ++i)
	{
		if (d[i] <= 1)
			k[i] = 0;
	}

	auto error = 0.5 * std::pow(Dot(k, mesh.GetDx()), 2.0);

	TOneForm f1(mesh);
	TZeroForm f0(mesh);

	f0.Clear();
	f1.Clear();

	for (auto s : mesh.GetRange(EDGE))
	{
		f1[s] = std::sin(Dot(k, mesh.GetCoordinates(s)));
	};

	f0 = Diverge(f1);

	Real variance = 0;
	value_type average = 0.0;

	for (auto s : mesh.GetRange(VERTEX))
	{

		auto expect = std::cos(Dot(k, mesh.GetCoordinates(s))) * (k[0] + k[1] + k[2]);

		variance += abs((f0[s] - expect) * (f0[s] - expect));

		average += (f0[s] - expect);

		auto x = mesh.GetCoordinates(s);
		if ((abs(f0[s]) > epsilon || abs(expect) > epsilon))
			ASSERT_LE(abs(2.0*(f0[s]-expect)/(f0[s] + expect)), error )<< expect / f0[s] << " " << f0[s] << " "
			<< expect << " " << (mesh.GetCoordinates(s));
		;

	}

	variance /= f0.size();
	average /= f0.size();
	CHECK(variance);
	CHECK(average);

}

TEST_P(TestDiffCalculus, diverge2)
{
	nTuple<3, Real> k = K;
	auto d = mesh.GetDimensions();

	for (int i = 0; i < mesh.NDIMS; ++i)
	{
		if (d[i] <= 1)
			k[i] = 0;
	}
	auto error = 0.5 * std::pow(Dot(k, mesh.GetDx()), 2.0);

	TTwoForm f2(mesh);
	TThreeForm f3(mesh);

	f3.Clear();
	f2.Clear();

	for (auto s : mesh.GetRange(FACE))
	{
		f2[s] = std::sin(Dot(k, mesh.GetCoordinates(s)));
	};

	f3 = Diverge(f2);

	Real variance = 0;
	value_type average = 0.0;

	for (auto s : mesh.GetRange(VOLUME))
	{

		auto expect = std::cos(Dot(k, mesh.GetCoordinates(s))) * (k[0] + k[1] + k[2]);

		variance += abs((f3[s] - expect) * (f3[s] - expect));

		average += (f3[s] - expect);

		if (abs(f3[s]) > epsilon || abs(expect) > epsilon)
			ASSERT_LE(abs(2.0*(f3[s]-expect)/(f3[s] + expect)), error )<< " " << expect / f3[s] << " " << f3[s]
			<< " " << expect;

		}

	variance /= f3.size();
	average /= f3.size();
	CHECK(variance);
	CHECK(average);

}

TEST_P(TestDiffCalculus, curl1)
{
	nTuple<3, Real> k = K;
	auto d = mesh.GetDimensions();
	CHECK(d);
	for (int i = 0; i < mesh.NDIMS; ++i)
	{
		if (d[i] <= 1)
			k[i] = 0;
	}
	auto error = std::pow(Dot(k, mesh.GetDx()), 2.0);

	TOneForm vf1(mesh);
	TOneForm vf1b(mesh);
	TTwoForm vf2(mesh);
	TTwoForm vf2b(mesh);

	vf1.Clear();
	vf1b.Clear();
	vf2.Clear();
	vf2b.Clear();

	Real m = 0.0;
	Real variance = 0;
	value_type average;
	average *= 0.0;

	for (auto s : mesh.GetRange(EDGE))
	{
		vf1[s] = std::sin(Dot(k, mesh.GetCoordinates(s)));
	};

	LOG_CMD(vf2 = Curl(vf1));

	for (auto s : mesh.GetRange(FACE))
	{
		auto n = mesh.ComponentNum(s.self_);

		auto expect = std::cos(Dot(k, mesh.GetCoordinates(s))) * (k[(n + 1) % 3] - k[(n + 2) % 3]);

		variance += abs((vf2[s] - expect) * (vf2[s] - expect));

		average += (vf2[s] - expect);
		auto x = mesh.GetCoordinates(s);
		if ((abs(vf2[s]) > epsilon || abs(expect) > epsilon))
			ASSERT_LE(abs(2.0*(vf2[s]-expect)/(vf2[s] + expect)), error )

<<			vf2[s] << " " << expect << " " << x<<" "<<mesh.GetExtents();

		}

	variance /= vf2.size();
	average /= vf2.size();
	CHECK(variance);
	CHECK(average);

}

TEST_P(TestDiffCalculus, curl2)
{
	nTuple<3, Real> k = K;
	auto d = mesh.GetDimensions();

	for (int i = 0; i < mesh.NDIMS; ++i)
	{
		if (d[i] <= 1)
			k[i] = 0;
	}
	auto error = std::pow(Dot(k, mesh.GetDx()), 2.0);

	TOneForm vf1(mesh);
	TOneForm vf1b(mesh);
	TTwoForm vf2(mesh);
	TTwoForm vf2b(mesh);

	vf1.Clear();
	vf1b.Clear();
	vf2.Clear();
	vf2b.Clear();

	Real m = 0.0;
	Real variance = 0;
	value_type average;
	average *= 0.0;

	for (auto s : mesh.GetRange(FACE))
	{
		vf2[s] = std::sin(Dot(k, mesh.GetCoordinates(s)));
	};

	LOG_CMD(vf1 = Curl(vf2));

	vf1b.Clear();

	for (auto s : mesh.GetRange(EDGE))
	{

		auto n = mesh.ComponentNum(s.self_);

		auto expect = std::cos(Dot(k, mesh.GetCoordinates(s))) * (k[(n + 1) % 3] - k[(n + 2) % 3]);

		vf1b[s] = expect;

		variance += abs((vf1[s] - expect) * (vf1[s] - expect));

		average += (vf1[s] - expect);

		auto x = mesh.GetCoordinates(s);
		if ((abs(vf1[s]) > epsilon || abs(expect) > epsilon))
			ASSERT_LE(abs(2.0*(vf1[s]-expect)/(vf1[s] + expect)), error )<< vf1[s] << " " << expect << " "
			<< mesh.GetCoordinates(s);

		}

	variance /= vf1.size();
	average /= vf1.size();
	CHECK(variance);
	CHECK(average);

}

TEST_P(TestDiffCalculus, identity_curl_grad_f0_eq_0)
{

	TZeroForm f0(mesh);

	TOneForm f1(mesh);
	TTwoForm f2a(mesh);
	TTwoForm f2b(mesh);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m = 0.0;
	f0.Clear();
	for (auto s : mesh.GetRange(VERTEX))
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

	for (auto s : mesh.GetRange(FACE))
	{

		relative_error += abs(f2b[s]);
		EXPECT_EQ((f2a[s]), (f2b[s]));
	}

	relative_error /= m;

	INFORM2(relative_error);
	ASSERT_LE(relative_error, epsilon)<<m;

}

TEST_P(TestDiffCalculus, identity_curl_grad_f3_eq_0)
{

	TThreeForm f3(mesh);
	TOneForm f1a(mesh);
	TOneForm f1b(mesh);
	TTwoForm f2(mesh);
	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m = 0.0;

	f3.Clear();

	for (auto s : mesh.GetRange(VOLUME))
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

	for (auto s : mesh.GetRange(EDGE))
	{

		EXPECT_EQ((f1a[s]), (f1b[s]));

		relative_error += abs(f1b[s]);

	}

	relative_error /= m;

	INFORM2(relative_error);
	ASSERT_LE(relative_error, epsilon);

}

TEST_P(TestDiffCalculus, identity_div_curl_f1_eq0)
{

	TOneForm f1(mesh);
	TTwoForm f2(mesh);
	TZeroForm f0a(mesh);
	TZeroForm f0b(mesh);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	f2.Clear();

	Real m = 0.0;

	for (auto s : mesh.GetRange(FACE))
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

	for (auto s : mesh.GetRange(VERTEX))
	{
		relative_error += abs(f0b[s]);
		EXPECT_EQ((f0a[s]), (f0b[s]));
	}

	relative_error /= m;
	INFORM2(relative_error);
	ASSERT_LE(relative_error, epsilon);

}

TEST_P(TestDiffCalculus, identity_div_curl_f2_eq0)
{

	TOneForm f1(mesh);
	TTwoForm f2(mesh);
	TThreeForm f3a(mesh);
	TThreeForm f3b(mesh);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	f1.Clear();

	Real m = 0.0;

	for (auto s : mesh.GetRange(EDGE))
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

	for (auto s : mesh.GetRange(VOLUME))
	{

		EXPECT_DOUBLE_EQ(abs(f3a[s]), abs(f3b[s]));

		relative_error += abs(f3b[s]);

	}

	relative_error /= m;
	INFORM2(relative_error);
	ASSERT_LE(relative_error, epsilon);

}

#endif /* FETL_TEST3_H_ */
