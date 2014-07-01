/*
 * fetl_test3.h
 *
 *  Created on: 2014年3月24日
 *      Author: salmon
 */

#ifndef FETL_TEST3_H_
#define FETL_TEST3_H_

#include <gtest/gtest.h>
#include "fetl_test.h"
#include "save_field.h"

using namespace simpla;

TEST_P(TestFETL, grad0)
{

	Real error = abs(std::pow(InnerProductNTuple(K_real, mesh.GetDx()), 2.0));

	auto f0 = mesh.make_field<VERTEX, scalar_type>();
	auto f1 = mesh.make_field<EDGE, scalar_type>();
	auto f1b = mesh.make_field<EDGE, scalar_type>();

	f0.clear();
	f1.clear();
	f1b.clear();

	for (auto s : mesh.Select(VERTEX))
	{
		f0[s] = std::sin(InnerProductNTuple(K_real, mesh.GetCoordinates(s)));
	};

	LOG_CMD(f1 = Grad(f0));

	Real m = 0.0;
	Real variance = 0;
	scalar_type average;
	average *= 0.0;

	for (auto s : mesh.Select(EDGE))
	{
		unsigned int n = mesh.ComponentNum(s);

		auto x = mesh.GetCoordinates(s);

		scalar_type expect = K_real[n] * std::cos(InnerProductNTuple(K_real, x))
		        + K_imag[n] * std::sin(InnerProductNTuple(K_real, x));

		if (mesh.TypeAsString() == "Cylindrical" && n == (mesh_type::ZAxis + 1) % 3)
		{
			auto r = mesh.GetCoordinates(s);
			expect /= r[(mesh_type::ZAxis + 2) % 3];
		}

		f1b[s] = expect;

		variance += abs((f1[s] - expect) * (f1[s] - expect));

		average += (f1[s] - expect);

//		if (abs(expect) > EPSILON)
//		{
//			EXPECT_LE(abs(2.0 * (f1[s] - expect) / (f1[s] + expect)), error) << " expect = " << expect
//			        << " actual = " << f1[s] << " x= " << mesh.GetCoordinates(s) << " K= " << K_real << " mesh.K="
//			        << mesh.k_imag;
//		}
//		else
//		{
//			EXPECT_LE(abs(f1[s]), error) << " expect = " << expect << " actual = " << f1[s] << " x= "
//			        << mesh.GetCoordinates(s);
//
//		}

//		if (abs(f1[s]) > epsilon || abs(expect) > epsilon)
//		{
//			ASSERT_GE(error, abs(2.0 * (f1[s] - expect) / (f1[s] + expect)));
//		}

	}
//	GLOBAL_DATA_STREAM.OpenGroup("/grad0/");
//	LOGGER << SAVE(f0);
//	LOGGER << SAVE(f1);
//	LOGGER << SAVE(f1b);
	variance /= mesh.GetNumOfElements(EDGE);
	average /= mesh.GetNumOfElements(EDGE);
	ASSERT_LE(std::sqrt(variance), error);
	ASSERT_LE(std::abs(average), error);
}

TEST_P(TestFETL, grad3)
{

	Real error = abs(std::pow(InnerProductNTuple(K_real, mesh.GetDx()), 2.0));

	auto f2 = mesh.make_field<FACE, scalar_type>();
	auto f2b = mesh.make_field<FACE, scalar_type>();
	auto f3 = mesh.make_field<VOLUME, scalar_type>();

	f3.clear();
	f2.clear();
	f2b.clear();

	for (auto s : mesh.Select(VOLUME))
	{
		f3[s] = std::sin(InnerProductNTuple(K_real, mesh.GetCoordinates(s)));
	};

	LOG_CMD(f2 = Grad(f3));

	Real m = 0.0;
	Real variance = 0;
	scalar_type average;
	average *= 0.0;

	for (auto s : mesh.Select(FACE))
	{

		unsigned int n = mesh.ComponentNum(s);

		auto x = mesh.GetCoordinates(s);

		scalar_type expect = K_real[n] * std::cos(InnerProductNTuple(K_real, x))
		        + K_imag[n] * std::sin(InnerProductNTuple(K_real, x));

		if (mesh.TypeAsString() == "Cylindrical" && n == (mesh_type::ZAxis + 1) % 3)
		{
			auto r = mesh.GetCoordinates(s);
			expect /= r[(mesh_type::ZAxis + 2) % 3];
		}

		f2b[s] = expect;

		variance += abs((f2[s] - expect) * (f2[s] - expect));

		average += (f2[s] - expect);

//		if (abs(expect) > EPSILON)
//		{
//			EXPECT_LE(abs(2.0 * (f2[s] - expect) / (f2[s] + expect)), error) << " expect = " << expect
//			        << " actual = " << f2[s] << " x= " << mesh.GetCoordinates(s) << " K= " << K_real;
//		}
//		else
//		{
//			EXPECT_LE(abs(f2[s]), error) << " expect = " << expect << " actual = " << f2[s] << " x= "
//			        << mesh.GetCoordinates(s);
//
//		}

	}

	variance /= f2.size();
	average /= f2.size();
	EXPECT_LE(std::sqrt(variance), error);
	EXPECT_LE(std::abs(average), error);

//	GLOBAL_DATA_STREAM.OpenGroup("/grad3/");
//	LOGGER << SAVE(f3);
//	LOGGER << SAVE(f2);
//	LOGGER << SAVE(f2b);

}

TEST_P(TestFETL, diverge1)
{

	Real error = abs(std::pow(InnerProductNTuple(K_real, mesh.GetDx()), 2.0));

	auto f1 = mesh.make_field<EDGE, scalar_type>();
	auto f0 = mesh.make_field<VERTEX, scalar_type>();
	auto f0b = mesh.make_field<VERTEX, scalar_type>();
	f0.clear();
	f0b.clear();
	f1.clear();

	for (auto s : mesh.Select(EDGE))
	{
		f1[s] = std::sin(InnerProductNTuple(K_real, mesh.GetCoordinates(s)));
	};

	f0 = Diverge(f1);

	Real variance = 0;

	scalar_type average = 0.0;

	for (auto s : mesh.Select(VERTEX))
	{

		auto x = mesh.GetCoordinates(s);

		Real cos_v = std::cos(InnerProductNTuple(K_real, x));
		Real sin_v = std::sin(InnerProductNTuple(K_real, x));

		scalar_type expect;

		if (mesh.TypeAsString() == "Cylindrical")
		{

			expect =

			(K_real[(mesh_type::ZAxis + 1) % 3] / x[(mesh_type::ZAxis + 2) % 3] + //  k_theta

			        K_real[(mesh_type::ZAxis + 2) % 3] + //  k_r

			        K_real[(mesh_type::ZAxis + 3) % 3] //  k_z
			) * cos_v

			+ (K_imag[(mesh_type::ZAxis + 1) % 3] / x[(mesh_type::ZAxis + 2) % 3] + //  k_theta

			        K_imag[(mesh_type::ZAxis + 2) % 3] + //  k_r

			        K_imag[(mesh_type::ZAxis + 3) % 3] //  k_z
			) * sin_v;

			expect += sin_v / x[(mesh_type::ZAxis + 2) % 3]; //A_r
		}
		else
		{
			expect = (K_real[0] + K_real[1] + K_real[2]) * cos_v + (K_imag[0] + K_imag[1] + K_imag[2]) * sin_v;
		}

		f0b[s] = expect;

		variance += abs((f0[s] - expect) * (f0[s] - expect));

		average += (f0[s] - expect);

//		if (abs(expect) > EPSILON)
//		{
//			EXPECT_LE(abs(2.0 * (f0[s] - expect) / (f0[s] + expect)), error) << " expect = " << expect
//			        << " actual = " << f0[s] << " x= " << mesh.GetCoordinates(s) << " K= " << K_real << " K_i= "
//			        << K_imag;
//		}
//		else
//		{
//			EXPECT_LE(abs(f0[s]), error) << " expect = " << expect << " actual = " << f0[s] << " x= "
//			        << mesh.GetCoordinates(s);
//
//		}
	}

	variance /= f0.size();
	average /= f0.size();

	EXPECT_LE(std::sqrt(variance), error);
	EXPECT_LE(std::abs(average), error) << " K= " << K_real << " K_i= " << K_imag << " mesh.Ki=" << mesh.k_imag;

//	GLOBAL_DATA_STREAM.OpenGroup("/diverge1/");
//	LOGGER << SAVE(f1);
//	LOGGER << SAVE(f0);
//	LOGGER << SAVE(f0b);

}

TEST_P(TestFETL, diverge2)
{

	Real error = abs(std::pow(InnerProductNTuple(K_real, mesh.GetDx()), 2.0));

	auto f2 = mesh.make_field<FACE, scalar_type>();
	auto f3 = mesh.make_field<VOLUME, scalar_type>();

	f3.clear();
	f2.clear();

	for (auto s : mesh.Select(FACE))
	{
		f2[s] = std::sin(InnerProductNTuple(K_real, mesh.GetCoordinates(s)));
	};

	f3 = Diverge(f2);

	Real variance = 0;
	scalar_type average = 0.0;

	for (auto s : mesh.Select(VOLUME))
	{
		auto x = mesh.GetCoordinates(s);

		Real cos_v = std::cos(InnerProductNTuple(K_real, x));
		Real sin_v = std::sin(InnerProductNTuple(K_real, x));

		scalar_type expect;

		if (mesh.TypeAsString() == "Cylindrical")
		{

			expect =

			(K_real[(mesh_type::ZAxis + 1) % 3] / x[(mesh_type::ZAxis + 2) % 3] + //  k_theta

			        K_real[(mesh_type::ZAxis + 2) % 3] + //  k_r

			        K_real[(mesh_type::ZAxis + 3) % 3] //  k_z
			) * cos_v

			+ (K_imag[(mesh_type::ZAxis + 1) % 3] / x[(mesh_type::ZAxis + 2) % 3] + //  k_theta

			        K_imag[(mesh_type::ZAxis + 2) % 3] + //  k_r

			        K_imag[(mesh_type::ZAxis + 3) % 3] //  k_z
			) * sin_v;

			expect += std::sin(InnerProductNTuple(K_real, mesh.GetCoordinates(s))) / x[(mesh_type::ZAxis + 2) % 3]; //A_r
		}
		else
		{
			expect = (K_real[0] + K_real[1] + K_real[2]) * cos_v + (K_imag[0] + K_imag[1] + K_imag[2]) * sin_v;
		}

		variance += abs((f3[s] - expect) * (f3[s] - expect));

		average += (f3[s] - expect);

//		if (abs(f3[s]) > epsilon || abs(expect) > epsilon)
//			ASSERT_LE(abs(2.0 * (f3[s] - expect) / (f3[s] + expect)), error);

	}

	variance /= f3.size();
	average /= f3.size();

	ASSERT_LE(std::sqrt(variance), error);
	ASSERT_LE(std::abs(average), error);

}

TEST_P(TestFETL, curl1)
{

	Real error = abs(std::pow(InnerProductNTuple(K_real, mesh.GetDx()), 2.0));

	auto vf1 = mesh.make_field<EDGE, scalar_type>();
	auto vf1b = mesh.make_field<EDGE, scalar_type>();
	auto vf2 = mesh.make_field<FACE, scalar_type>();
	auto vf2b = mesh.make_field<FACE, scalar_type>();

	vf1.clear();
	vf1b.clear();
	vf2.clear();
	vf2b.clear();

	Real m = 0.0;
	Real variance = 0;
	scalar_type average;
	average *= 0.0;

	for (auto s : mesh.Select(EDGE))
	{
		vf1[s] = std::sin(InnerProductNTuple(K_real, mesh.GetCoordinates(s)));
	};

	LOG_CMD(vf2 = Curl(vf1));

	for (auto s : mesh.Select(FACE))
	{
		auto n = mesh.ComponentNum(s);

		auto x = mesh.GetCoordinates(s);

		Real cos_v = std::cos(InnerProductNTuple(K_real, x));
		Real sin_v = std::sin(InnerProductNTuple(K_real, x));

		scalar_type expect;

		if (mesh.TypeAsString() == "Cylindrical")
		{
			switch (n)
			{
			case (mesh_type::ZAxis + 1) % 3: // theta
				expect = (K_real[(mesh_type::ZAxis + 2) % 3] - K_real[(mesh_type::ZAxis + 3) % 3]) * cos_v
				        + (K_imag[(mesh_type::ZAxis + 2) % 3] - K_imag[(mesh_type::ZAxis + 3) % 3]) * sin_v;
				break;
			case (mesh_type::ZAxis + 2) % 3: // r
				expect = (K_real[(mesh_type::ZAxis + 3) % 3]
				        - K_real[(mesh_type::ZAxis + 1) % 3] / x[(mesh_type::ZAxis + 2) % 3]) * cos_v

				        + (K_imag[(mesh_type::ZAxis + 3) % 3]
				                - K_imag[(mesh_type::ZAxis + 1) % 3] / x[(mesh_type::ZAxis + 2) % 3]) * sin_v;
				break;

			case (mesh_type::ZAxis + 3) % 3: // z
				expect = (K_real[(mesh_type::ZAxis + 1) % 3] - K_real[(mesh_type::ZAxis + 2) % 3]) * cos_v
				        + (K_imag[(mesh_type::ZAxis + 1) % 3] - K_imag[(mesh_type::ZAxis + 2) % 3]) * sin_v;

				expect -= std::sin(InnerProductNTuple(K_real, mesh.GetCoordinates(s))) / x[(mesh_type::ZAxis + 2) % 3]; //A_r
				break;

			}

		}
		else
		{
			expect = (K_real[(n + 1) % 3] - K_real[(n + 2) % 3]) * cos_v
			        + (K_imag[(n + 1) % 3] - K_imag[(n + 2) % 3]) * sin_v;
		}

		vf2b[s] = expect;

		variance += abs((vf2[s] - expect) * (vf2[s] - expect));

		average += (vf2[s] - expect);

//		if (abs(expect) > epsilon)
//		{
//			EXPECT_LE(abs(2.0 * (vf2[s] - expect) / (vf2[s] + expect)), error) << " expect = " << expect
//			        << " actual = " << vf2[s] << " x= " << mesh.GetCoordinates(s);
//		}
//		else
//		{
//			EXPECT_LE(abs(vf2[s]), error) << " expect = " << expect << " actual = " << vf2[s] << " x= "
//			        << mesh.GetCoordinates(s);
//		}

	}

	variance /= vf2.size();
	average /= vf2.size();

	ASSERT_LE(std::sqrt(variance), error);
	ASSERT_LE(std::abs(average), error);

}

TEST_P(TestFETL, curl2)
{
	Real error = abs(std::pow(InnerProductNTuple(K_real, mesh.GetDx()), 2.0));

	auto vf1 = mesh.make_field<EDGE, scalar_type>();
	auto vf1b = mesh.make_field<EDGE, scalar_type>();
	auto vf2 = mesh.make_field<FACE, scalar_type>();
	auto vf2b = mesh.make_field<FACE, scalar_type>();

	vf1.clear();
	vf1b.clear();
	vf2.clear();
	vf2b.clear();

	Real m = 0.0;
	Real variance = 0;
	scalar_type average;
	average *= 0.0;

	for (auto s : mesh.Select(FACE))
	{
		vf2[s] = std::sin(InnerProductNTuple(K_real, mesh.GetCoordinates(s)));
	};

	LOG_CMD(vf1 = Curl(vf2));

	vf1b.clear();

	for (auto s : mesh.Select(EDGE))
	{

		auto n = mesh.ComponentNum(s);

		auto x = mesh.GetCoordinates(s);

		Real cos_v = std::cos(InnerProductNTuple(K_real, x));
		Real sin_v = std::sin(InnerProductNTuple(K_real, x));

		scalar_type expect;

		if (mesh.TypeAsString() == "Cylindrical")
		{
			switch (n)
			{
			case (mesh_type::ZAxis + 1) % 3: // theta
				expect = (K_real[(mesh_type::ZAxis + 2) % 3] - K_real[(mesh_type::ZAxis + 3) % 3]) * cos_v
				        + (K_imag[(mesh_type::ZAxis + 2) % 3] - K_imag[(mesh_type::ZAxis + 3) % 3]) * sin_v;
				break;
			case (mesh_type::ZAxis + 2) % 3: // r
				expect = (K_real[(mesh_type::ZAxis + 3) % 3]
				        - K_real[(mesh_type::ZAxis + 1) % 3] / x[(mesh_type::ZAxis + 2) % 3]) * cos_v

				        + (K_imag[(mesh_type::ZAxis + 3) % 3]
				                - K_imag[(mesh_type::ZAxis + 1) % 3] / x[(mesh_type::ZAxis + 2) % 3]) * sin_v;
				break;

			case (mesh_type::ZAxis + 3) % 3: // z
				expect = (K_real[(mesh_type::ZAxis + 1) % 3] - K_real[(mesh_type::ZAxis + 2) % 3]) * cos_v
				        + (K_imag[(mesh_type::ZAxis + 1) % 3] - K_imag[(mesh_type::ZAxis + 2) % 3]) * sin_v;

				expect -= std::sin(InnerProductNTuple(K_real, mesh.GetCoordinates(s))) / x[(mesh_type::ZAxis + 2) % 3]; //A_r
				break;

			}

		}
		else
		{
			expect = (K_real[(n + 1) % 3] - K_real[(n + 2) % 3]) * cos_v
			        + (K_imag[(n + 1) % 3] - K_imag[(n + 2) % 3]) * sin_v;
		}

		vf1b[s] = expect;

		variance += abs((vf1[s] - expect) * (vf1[s] - expect));

		average += (vf1[s] - expect);

//		if (abs(expect) > epsilon)
//		{
//			ASSERT_LE(abs(2.0 * (vf1[s] - expect) / (vf1[s] + expect)), error)<< " expect = "<<expect<<" actual = "<<vf1[s]<< " x= "<<mesh.GetCoordinates(s);
//		}
//		else
//		{
//			ASSERT_LE(abs(vf1[s]), error)<< " expect = "<<expect<<" actual = "<<vf1[s]<< " x= "<<mesh.GetCoordinates(s);
//
//		}

	}

	variance /= vf1.size();
	average /= vf1.size();

	ASSERT_LE(std::sqrt(variance), error);
	ASSERT_LE(std::abs(average), error);

}

TEST_P(TestFETL, identity_curl_grad_f0_eq_0)
{

	Real error = abs(std::pow(InnerProductNTuple(K_real, mesh.GetDx()), 2.0));

	auto f0 = mesh.make_field<VERTEX, scalar_type>();
	auto f1 = mesh.make_field<EDGE, scalar_type>();
	auto f2a = mesh.make_field<FACE, scalar_type>();
	auto f2b = mesh.make_field<FACE, scalar_type>();

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m = 0.0;
	f0.clear();
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
	Real variance_a = 0;
	Real variance_b = 0;
	for (auto s : mesh.Select(FACE))
	{

		variance_a += abs(f2a[s]);
		variance_b += abs(f2b[s]);
//		ASSERT_EQ((f2a[s]), (f2b[s]));
	}

	variance_a /= m;
	variance_b /= m;
	ASSERT_LE(std::sqrt(variance_b), error);
	ASSERT_LE(std::sqrt(variance_a), error);

}

TEST_P(TestFETL, identity_curl_grad_f3_eq_0)
{
	Real error = abs(std::pow(InnerProductNTuple(K_real, mesh.GetDx()), 2.0));

	auto f3 = mesh.make_field<VOLUME, scalar_type>();
	auto f1a = mesh.make_field<EDGE, scalar_type>();
	auto f1b = mesh.make_field<EDGE, scalar_type>();
	auto f2 = mesh.make_field<FACE, scalar_type>();
	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m = 0.0;

	f3.clear();

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
	Real variance_a = 0;
	Real variance_b = 0;
	for (auto s : mesh.Select(EDGE))
	{

//		ASSERT_EQ((f1a[s]), (f1b[s]));
		variance_a += abs(f1a[s]);
		variance_b += abs(f1b[s]);

	}

	variance_a /= m;
	variance_b /= m;
	ASSERT_LE(std::sqrt(variance_b), error);
	ASSERT_LE(std::sqrt(variance_a), error);
}

TEST_P(TestFETL, identity_div_curl_f1_eq0)
{
	Real error = abs(std::pow(InnerProductNTuple(K_real, mesh.GetDx()), 2.0));

	auto f1 = mesh.make_field<EDGE, scalar_type>();
	auto f2 = mesh.make_field<FACE, scalar_type>();
	auto f0a = mesh.make_field<VERTEX, scalar_type>();
	auto f0b = mesh.make_field<VERTEX, scalar_type>();

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	f2.clear();

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
	Real variance_a = 0;
	Real variance_b = 0;
	for (auto s : mesh.Select(VERTEX))
	{
		variance_b += abs(f0b[s] * f0b[s]);
		variance_a += abs(f0a[s] * f0a[s]);
//		ASSERT_EQ((f0a[s]), (f0b[s]));
	}
	variance_a /= m;
	variance_b /= m;
	ASSERT_LE(std::sqrt(variance_b), error);
	ASSERT_LE(std::sqrt(variance_a), error);

}

TEST_P(TestFETL, identity_div_curl_f2_eq0)
{
	Real error = abs(std::pow(InnerProductNTuple(K_real, mesh.GetDx()), 2.0));

	auto f1 = mesh.make_field<EDGE, scalar_type>();
	auto f2 = mesh.make_field<FACE, scalar_type>();
	auto f3a = mesh.make_field<VOLUME, scalar_type>();
	auto f3b = mesh.make_field<VOLUME, scalar_type>();

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	f1.clear();

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

	Real variance_a = 0;
	Real variance_b = 0;
	for (auto s : mesh.Select(VOLUME))
	{

//		ASSERT_DOUBLE_EQ(abs(f3a[s]), abs(f3b[s]));
		variance_a += abs(f3a[s] * f3a[s]);
		variance_b += abs(f3b[s] * f3b[s]);

	}

	variance_a /= m;
	variance_b /= m;
	ASSERT_LE(std::sqrt(variance_b), error);
	ASSERT_LE(std::sqrt(variance_a), error);

}

#endif /* FETL_TEST3_H_ */
