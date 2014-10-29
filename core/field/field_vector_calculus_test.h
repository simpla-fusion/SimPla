/*
 * field_vector_calculus_test.h
 *
 *  Created on: 2014年10月21日
 *      Author: salmon
 */

#ifndef CORE_FIELD_FIELD_VECTOR_CALCULUS_TEST_H_
#define CORE_FIELD_FIELD_VECTOR_CALCULUS_TEST_H_

#include <gtest/gtest.h>
#include <tuple>

#include "../utilities/log.h"
#include "../utilities/ntuple.h"
#include "../utilities/pretty_stream.h"
#include "../io/data_stream.h"
#include "field.h"
#include "save_field.h"
#include "update_ghosts_field.h"

using namespace simpla;

#ifndef TMESH
#include "../manifold/manifold.h"

#include "../manifold/geometry/cartesian.h"
#include "../manifold/topology/structured.h"
#include "../manifold/diff_scheme/fdm.h"
#include "../manifold/interpolator/interpolator.h"

#define TMESH Manifold<CartesianCoordinates<StructuredMesh, CARTESIAN_ZAXIS>,FiniteDiffMethod, InterpolatorLinear> ;

#endif

typedef TMESH TManifold;

class TestFETL: public testing::TestWithParam<
		std::tuple<nTuple<Real, 3>, nTuple<Real, 3>, nTuple<size_t, 3>,
				nTuple<Real, 3>> >
{

protected:
	void SetUp()
	{
		LOGGER.set_stdout_visable_level(LOG_VERBOSE);

		std::tie(xmin, xmax, dims, K_real) = GetParam();
		SetDefaultValue(&default_value);

		for (int i = 0; i < ndims; ++i)
		{
			if (dims[i] <= 1 || (xmax[i] <= xmin[i]))
			{
				dims[i] = 1;
				K_real[i] = 0.0;
				xmax[i] = xmin[i];
			}
		}

		mesh.dimensions(dims);
		mesh.extents(xmin, xmax);
		mesh.update();

		Vec3 dx = mesh.dx();

		error = std::pow(inner_product(K_real, dx), 2.0);
//		for (int i = 0; i < ndims; ++i)
//		{
//			error += std::pow((K_real[i] * dx[i]), 2.0);
//		}
	}
public:

	typedef TManifold manifold_type;
	typedef Real value_type;
	typedef/*typename manifold_type::scalar_type*/Real scalar_type;
	typedef typename manifold_type::iterator iterator;
	typedef typename manifold_type::coordinates_type coordinates_type;

	manifold_type mesh;

	static constexpr size_t ndims = manifold_type::ndims;

	nTuple<Real, 3> xmin;

	nTuple<Real, 3> xmax;

	nTuple<size_t, 3> dims;

	nTuple<Real, 3> K_real;	// @NOTE must   k = n TWOPI, period condition

	nTuple<scalar_type, 3> K_imag;

	value_type default_value;

	Real error;

	template<typename T>
	void SetDefaultValue(T* v)
	{
		*v = 1;
	}
	template<typename T>
	void SetDefaultValue(std::complex<T>* v)
	{
		T r;
		SetDefaultValue(&r);
		*v = std::complex<T>();
	}

	template<size_t N, typename T>
	void SetDefaultValue(nTuple<T, N>* v)
	{
		for (int i = 0; i < N; ++i)
		{
			(*v)[i] = i;
		}
	}

	virtual ~TestFETL()
	{

	}

};

TEST_P(TestFETL, grad0)
{
	auto domain0 = make_domain<VERTEX>(mesh);
	auto domain1 = make_domain<EDGE>(mesh);
	auto domain2 = make_domain<FACE>(mesh);
	auto domain3 = make_domain<VOLUME>(mesh);

	auto f0 = make_field<scalar_type>(domain0);
	auto f1 = make_field<scalar_type>(domain1);
	auto f1b = make_field<scalar_type>(domain1);

	f0.clear();
	f1.clear();
	f1b.clear();

	for (auto s : domain0)
	{
		f0[s] = std::sin(inner_product(K_real, mesh.coordinates(s)));
	};
	update_ghosts(&f0);

	LOG_CMD(f1 = grad(f0));
	Real m = 0.0;
	Real variance = 0;
	scalar_type average;
	average *= 0.0;

	for (auto s : domain1)
	{
		size_t n = mesh.component_number(s);

		auto x = mesh.coordinates(s);

		scalar_type expect = K_real[n] * std::cos(inner_product(K_real, x))
				+ K_imag[n] * std::sin(inner_product(K_real, x));

		if (mesh.get_type_as_string() == "Cylindrical"
				&& n == (manifold_type::ZAxis + 1) % 3)
		{
			auto r = mesh.coordinates(s);
			expect /= r[(manifold_type::ZAxis + 2) % 3];
		}

		f1b[s] = expect;

		variance += abs((f1[s] - expect) * (f1[s] - expect));

		average += (f1[s] - expect);

//		if (abs(expect) > EPSILON)
//		{
//			EXPECT_LE(abs(2.0 * (f1[s] - expect) / (f1[s] + expect)), error) << " expect = " << expect
//			        << " actual = " << f1[s] << " x= " << mesh.coordinates(s) << " K= " << K_real << " mesh.K="
//			        << mesh.k_imag;
//		}
//		else
//		{
//			EXPECT_LE(abs(f1[s]), error) << " expect = " << expect << " actual = " << f1[s] << " x= "
//			        << mesh.coordinates(s);
//
//		}

//		if (abs(f1[s]) > epsilon || abs(expect) > epsilon)
//		{
//			ASSERT_GE(error, abs(2.0 * (f1[s] - expect) / (f1[s] + expect)));
//		}

	}
	GLOBAL_DATA_STREAM.cd("/grad0/");
	LOGGER << SAVE(f0);
	LOGGER << SAVE(f1);
	LOGGER << SAVE(f1b);
	variance /= mesh.get_num_of_elements(EDGE);
	average /= mesh.get_num_of_elements(EDGE);
	ASSERT_LE(std::sqrt(variance), error);
	ASSERT_LE(std::abs(average), error);
}
//
//TEST_P(TestFETL, grad3)
//{
//	if (!mesh.is_valid())
//		return;
//
//	auto domain3 = make_domain<VOLUME>(mesh);
//	auto domain2 = make_domain<FACE>(mesh);
//	auto f2 = make_field<scalar_type>(make_domain<FACE>(mesh));
//	auto f2b = make_field<scalar_type>(make_domain<FACE>(mesh));
//	auto f3 = make_field<scalar_type>(make_domain<VOLUME>(mesh));
//
//	f3.clear();
//	f2.clear();
//	f2b.clear();
//
//	for (auto s : domain3)
//	{
//		f3[s] = std::sin(inner_product(K_real, mesh.coordinates(s)));
//	};
//	update_ghosts(&f3);
//	LOG_CMD(f2 = grad(f3));
//
//	Real m = 0.0;
//	Real variance = 0;
//	scalar_type average;
//	average *= 0.0;
//
//	for (auto s : domain2)
//	{
//
//		size_t n = mesh.component_number(s);
//
//		auto x = mesh.coordinates(s);
//
//		scalar_type expect = K_real[n] * std::cos(inner_product(K_real, x))
//				+ K_imag[n] * std::sin(inner_product(K_real, x));
//
//		if (mesh.get_type_as_string() == "Cylindrical"
//				&& n == (manifold_type::ZAxis + 1) % 3)
//		{
//			auto r = mesh.coordinates(s);
//			expect /= r[(manifold_type::ZAxis + 2) % 3];
//		}
//
//		f2b[s] = expect;
//
//		variance += abs((f2[s] - expect) * (f2[s] - expect));
//
//		average += (f2[s] - expect);
//
////		if (abs(expect) > EPSILON)
////		{
////			EXPECT_LE(abs(2.0 * (f2[s] - expect) / (f2[s] + expect)), error) << " expect = " << expect
////			        << " actual = " << f2[s] << " x= " << mesh.coordinates(s) << " K= " << K_real;
////		}
////		else
////		{
////			EXPECT_LE(abs(f2[s]), error) << " expect = " << expect << " actual = " << f2[s] << " x= "
////			        << mesh.coordinates(s);
////
////		}
//
//	}
//
//	variance /= f2.size();
//	average /= f2.size();
//	EXPECT_LE(std::sqrt(variance), error) << dims;
//	EXPECT_LE(std::abs(average), error);
//
////	GLOBAL_DATA_STREAM.cd("/grad3/");
////	LOGGER << SAVE(f3);
////	LOGGER << SAVE(f2);
////	LOGGER << SAVE(f2b);
//
//}
//
//TEST_P(TestFETL, diverge1)
//{
//	if (!mesh.is_valid())
//		return;
//
//	auto domain0 = make_domain<VERTEX>(mesh);
//	auto domain1 = make_domain<EDGE>(mesh);
//	auto f1 = make_field<scalar_type>(domain1);
//	auto f0 = make_field<scalar_type>(domain0);
//	auto f0b = make_field<scalar_type>(domain0);
//	f0.clear();
//	f0b.clear();
//	f1.clear();
//
//	for (auto s : domain1)
//	{
//		f1[s] = std::sin(inner_product(K_real, mesh.coordinates(s)));
//	};
//	update_ghosts(&f1);
//	f0 = diverge(f1);
//
//	Real variance = 0;
//
//	scalar_type average = 0.0;
//
//	for (auto s : domain0)
//	{
//
//		auto x = mesh.coordinates(s);
//
//		Real cos_v = std::cos(inner_product(K_real, x));
//		Real sin_v = std::sin(inner_product(K_real, x));
//
//		scalar_type expect;
//
//		if (mesh.get_type_as_string() == "Cylindrical")
//		{
//
//			expect =
//
//			(K_real[(manifold_type::ZAxis + 1) % 3]
//					/ x[(manifold_type::ZAxis + 2) % 3] + //  k_theta
//
//					K_real[(manifold_type::ZAxis + 2) % 3] + //  k_r
//
//					K_real[(manifold_type::ZAxis + 3) % 3] //  k_z
//			) * cos_v
//
//					+ (K_imag[(manifold_type::ZAxis + 1) % 3]
//							/ x[(manifold_type::ZAxis + 2) % 3] + //  k_theta
//
//							K_imag[(manifold_type::ZAxis + 2) % 3] + //  k_r
//
//							K_imag[(manifold_type::ZAxis + 3) % 3] //  k_z
//					) * sin_v;
//
//			expect += sin_v / x[(manifold_type::ZAxis + 2) % 3]; //A_r
//		}
//		else
//		{
//			expect = (K_real[0] + K_real[1] + K_real[2]) * cos_v
//					+ (K_imag[0] + K_imag[1] + K_imag[2]) * sin_v;
//		}
//
//		f0b[s] = expect;
//
//		variance += abs((f0[s] - expect) * (f0[s] - expect));
//
//		average += (f0[s] - expect);
//
////		if (abs(expect) > EPSILON)
////		{
////			EXPECT_LE(abs(2.0 * (f0[s] - expect) / (f0[s] + expect)), error) << " expect = " << expect
////			        << " actual = " << f0[s] << " x= " << mesh.coordinates(s) << " K= " << K_real << " K_i= "
////			        << K_imag;
////		}
////		else
////		{
////			EXPECT_LE(abs(f0[s]), error) << " expect = " << expect << " actual = " << f0[s] << " x= "
////			        << mesh.coordinates(s);
////
////		}
//	}
//
//	variance /= f0.size();
//	average /= f0.size();
//
//	EXPECT_LE(std::sqrt(variance), error) << dims;
//	EXPECT_LE(std::abs(average), error) << " K= " << K_real << " K_i= "
//			<< K_imag
//
////			<< " mesh.Ki=" << mesh.k_imag
//
//			;
//
//}
//
//TEST_P(TestFETL, diverge2)
//{
//	if (!mesh.is_valid())
//		return;
//
//	auto domain3 = make_domain<VOLUME>(mesh);
//	auto domain2 = make_domain<FACE>(mesh);
//	auto f2 = make_field<scalar_type>(domain2);
//	auto f3 = make_field<scalar_type>(domain3);
//
//	f3.clear();
//	f2.clear();
//
//	for (auto s : domain2)
//	{
//		f2[s] = std::sin(inner_product(K_real, mesh.coordinates(s)));
//	};
//	update_ghosts(&f2);
//	f3 = diverge(f2);
//
//	Real variance = 0;
//	scalar_type average = 0.0;
//
//	for (auto s : domain3)
//	{
//		auto x = mesh.coordinates(s);
//
//		Real cos_v = std::cos(inner_product(K_real, x));
//		Real sin_v = std::sin(inner_product(K_real, x));
//
//		scalar_type expect;
//
//		if (mesh.get_type_as_string() == "Cylindrical")
//		{
//
//			expect =
//
//			(K_real[(manifold_type::ZAxis + 1) % 3]
//					/ x[(manifold_type::ZAxis + 2) % 3] + //  k_theta
//
//					K_real[(manifold_type::ZAxis + 2) % 3] + //  k_r
//
//					K_real[(manifold_type::ZAxis + 3) % 3] //  k_z
//			) * cos_v
//
//					+ (K_imag[(manifold_type::ZAxis + 1) % 3]
//							/ x[(manifold_type::ZAxis + 2) % 3] + //  k_theta
//
//							K_imag[(manifold_type::ZAxis + 2) % 3] + //  k_r
//
//							K_imag[(manifold_type::ZAxis + 3) % 3] //  k_z
//					) * sin_v;
//
//			expect += std::sin(inner_product(K_real, mesh.coordinates(s)))
//					/ x[(manifold_type::ZAxis + 2) % 3]; //A_r
//		}
//		else
//		{
//			expect = (K_real[0] + K_real[1] + K_real[2]) * cos_v
//					+ (K_imag[0] + K_imag[1] + K_imag[2]) * sin_v;
//		}
//
//		variance += abs((f3[s] - expect) * (f3[s] - expect));
//
//		average += (f3[s] - expect);
//
////		if (abs(f3[s]) > epsilon || abs(expect) > epsilon)
////			ASSERT_LE(abs(2.0 * (f3[s] - expect) / (f3[s] + expect)), error);
//
//	}
//
//	variance /= f3.size();
//	average /= f3.size();
//
//	ASSERT_LE(std::sqrt(variance), error);
//	ASSERT_LE(std::abs(average), error);
//
//}
//
//TEST_P(TestFETL, curl1)
//{
//	if (!mesh.is_valid())
//		return;
//
//	auto domain1 = make_domain<EDGE>(mesh);
//	auto domain2 = make_domain<FACE>(mesh);
//	auto vf1 = make_field<scalar_type>(domain1);
//	auto vf1b = make_field<scalar_type>(domain1);
//	auto vf2 = make_field<scalar_type>(domain2);
//	auto vf2b = make_field<scalar_type>(domain2);
//
//	vf1.clear();
//	vf1b.clear();
//	vf2.clear();
//	vf2b.clear();
//
//	Real m = 0.0;
//	Real variance = 0;
//	scalar_type average;
//	average *= 0.0;
//
//	for (auto s : domain1)
//	{
//		vf1[s] = std::sin(inner_product(K_real, mesh.coordinates(s)));
//	};
//	update_ghosts(&vf1);
//	LOG_CMD(vf2 = curl(vf1));
//
//	for (auto s : domain2)
//	{
//		auto n = mesh.component_number(s);
//
//		auto x = mesh.coordinates(s);
//
//		Real cos_v = std::cos(inner_product(K_real, x));
//		Real sin_v = std::sin(inner_product(K_real, x));
//
//		scalar_type expect;
//
//		if (mesh.get_type_as_string() == "Cylindrical")
//		{
//			switch (n)
//			{
//			case (manifold_type::ZAxis + 1) % 3: // theta
//				expect = (K_real[(manifold_type::ZAxis + 2) % 3]
//						- K_real[(manifold_type::ZAxis + 3) % 3]) * cos_v
//						+ (K_imag[(manifold_type::ZAxis + 2) % 3]
//								- K_imag[(manifold_type::ZAxis + 3) % 3])
//								* sin_v;
//				break;
//			case (manifold_type::ZAxis + 2) % 3: // r
//				expect = (K_real[(manifold_type::ZAxis + 3) % 3]
//						- K_real[(manifold_type::ZAxis + 1) % 3]
//								/ x[(manifold_type::ZAxis + 2) % 3]) * cos_v
//
//						+ (K_imag[(manifold_type::ZAxis + 3) % 3]
//								- K_imag[(manifold_type::ZAxis + 1) % 3]
//										/ x[(manifold_type::ZAxis + 2) % 3])
//								* sin_v;
//				break;
//
//			case (manifold_type::ZAxis + 3) % 3: // z
//				expect = (K_real[(manifold_type::ZAxis + 1) % 3]
//						- K_real[(manifold_type::ZAxis + 2) % 3]) * cos_v
//						+ (K_imag[(manifold_type::ZAxis + 1) % 3]
//								- K_imag[(manifold_type::ZAxis + 2) % 3])
//								* sin_v;
//
//				expect -= std::sin(inner_product(K_real, mesh.coordinates(s)))
//						/ x[(manifold_type::ZAxis + 2) % 3]; //A_r
//				break;
//
//			}
//
//		}
//		else
//		{
//			expect = (K_real[(n + 1) % 3] - K_real[(n + 2) % 3]) * cos_v
//					+ (K_imag[(n + 1) % 3] - K_imag[(n + 2) % 3]) * sin_v;
//		}
//
//		vf2b[s] = expect;
//
//		variance += abs((vf2[s] - expect) * (vf2[s] - expect));
//
//		average += (vf2[s] - expect);
//
////		if (abs(expect) > epsilon)
////		{
////			EXPECT_LE(abs(2.0 * (vf2[s] - expect) / (vf2[s] + expect)), error) << " expect = " << expect
////			        << " actual = " << vf2[s] << " x= " << mesh.coordinates(s);
////		}
////		else
////		{
////			EXPECT_LE(abs(vf2[s]), error) << " expect = " << expect << " actual = " << vf2[s] << " x= "
////			        << mesh.coordinates(s);
////		}
//
//	}
//
//	variance /= vf2.size();
//	average /= vf2.size();
//
//	ASSERT_LE(std::sqrt(variance), error);
//	ASSERT_LE(std::abs(average), error);
//
//}
//
//TEST_P(TestFETL, curl2)
//{
//	if (!mesh.is_valid())
//		return;
//
//	auto domain1 = make_domain<EDGE>(mesh);
//	auto domain2 = make_domain<FACE>(mesh);
//	auto vf1 = make_field<scalar_type>(domain1);
//	auto vf1b = make_field<scalar_type>(domain1);
//	auto vf2 = make_field<scalar_type>(domain2);
//	auto vf2b = make_field<scalar_type>(domain2);
//
//	vf1.clear();
//	vf1b.clear();
//	vf2.clear();
//	vf2b.clear();
//
//	Real m = 0.0;
//	Real variance = 0;
//	scalar_type average;
//	average *= 0.0;
//
//	for (auto s : domain2)
//	{
//		vf2[s] = std::sin(inner_product(K_real, mesh.coordinates(s)));
//	};
//	update_ghosts(&vf2);
//	LOG_CMD(vf1 = curl(vf2));
//
//	vf1b.clear();
//
//	for (auto s : domain1)
//	{
//
//		auto n = mesh.component_number(s);
//
//		auto x = mesh.coordinates(s);
//
//		Real cos_v = std::cos(inner_product(K_real, x));
//		Real sin_v = std::sin(inner_product(K_real, x));
//
//		scalar_type expect;
//
//		if (mesh.get_type_as_string() == "Cylindrical")
//		{
//			switch (n)
//			{
//			case (manifold_type::ZAxis + 1) % 3: // theta
//				expect = (K_real[(manifold_type::ZAxis + 2) % 3]
//						- K_real[(manifold_type::ZAxis + 3) % 3]) * cos_v
//						+ (K_imag[(manifold_type::ZAxis + 2) % 3]
//								- K_imag[(manifold_type::ZAxis + 3) % 3])
//								* sin_v;
//				break;
//			case (manifold_type::ZAxis + 2) % 3: // r
//				expect = (K_real[(manifold_type::ZAxis + 3) % 3]
//						- K_real[(manifold_type::ZAxis + 1) % 3]
//								/ x[(manifold_type::ZAxis + 2) % 3]) * cos_v
//
//						+ (K_imag[(manifold_type::ZAxis + 3) % 3]
//								- K_imag[(manifold_type::ZAxis + 1) % 3]
//										/ x[(manifold_type::ZAxis + 2) % 3])
//								* sin_v;
//				break;
//
//			case (manifold_type::ZAxis + 3) % 3: // z
//				expect = (K_real[(manifold_type::ZAxis + 1) % 3]
//						- K_real[(manifold_type::ZAxis + 2) % 3]) * cos_v
//						+ (K_imag[(manifold_type::ZAxis + 1) % 3]
//								- K_imag[(manifold_type::ZAxis + 2) % 3])
//								* sin_v;
//
//				expect -= std::sin(inner_product(K_real, mesh.coordinates(s)))
//						/ x[(manifold_type::ZAxis + 2) % 3]; //A_r
//				break;
//
//			}
//
//		}
//		else
//		{
//			expect = (K_real[(n + 1) % 3] - K_real[(n + 2) % 3]) * cos_v
//					+ (K_imag[(n + 1) % 3] - K_imag[(n + 2) % 3]) * sin_v;
//		}
//
//		vf1b[s] = expect;
//
//		variance += abs((vf1[s] - expect) * (vf1[s] - expect));
//
//		average += (vf1[s] - expect);
//
////		if (abs(expect) > epsilon)
////		{
////			ASSERT_LE(abs(2.0 * (vf1[s] - expect) / (vf1[s] + expect)), error)<< " expect = "<<expect<<" actual = "<<vf1[s]<< " x= "<<mesh.coordinates(s);
////		}
////		else
////		{
////			ASSERT_LE(abs(vf1[s]), error)<< " expect = "<<expect<<" actual = "<<vf1[s]<< " x= "<<mesh.coordinates(s);
////
////		}
//
//	}
//
//	variance /= vf1.size();
//	average /= vf1.size();
//
//	ASSERT_LE(std::sqrt(variance), error);
//	ASSERT_LE(std::abs(average), error);
//
//}
//
//TEST_P(TestFETL, identity_curl_grad_f0_eq_0)
//{
//	if (!mesh.is_valid())
//		return;
//	auto domain0 = make_domain<VERTEX>(mesh);
//	auto domain1 = make_domain<EDGE>(mesh);
//	auto domain2 = make_domain<FACE>(mesh);
//	auto domain3 = make_domain<VOLUME>(mesh);
//	auto f0 = make_field<scalar_type>(domain0);
//	auto f1 = make_field<scalar_type>(domain1);
//	auto f2a = make_field<scalar_type>(domain2);
//	auto f2b = make_field<scalar_type>(domain2);
//
//	std::mt19937 gen;
//	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);
//
//	Real m = 0.0;
//	f0.clear();
//	for (auto s : domain0)
//	{
//
//		auto a = uniform_dist(gen);
//		f0[s] = default_value * a;
//		m += a * a;
//	}
//	update_ghosts(&f0);
//
//	m = std::sqrt(m) * abs(default_value);
//
//	LOG_CMD(f1 = grad(f0));
//	LOG_CMD(f2a = curl(f1));
//	LOG_CMD(f2b = curl(grad(f0)));
//
//	size_t count = 0;
//	Real variance_a = 0;
//	Real variance_b = 0;
//	for (auto s : domain2)
//	{
//
//		variance_a += abs(f2a[s]);
//		variance_b += abs(f2b[s]);
////		ASSERT_EQ((f2a[s]), (f2b[s]));
//	}
//
//	variance_a /= m;
//	variance_b /= m;
//	ASSERT_LE(std::sqrt(variance_b), error);
//	ASSERT_LE(std::sqrt(variance_a), error);
//
//}
//
//TEST_P(TestFETL, identity_curl_grad_f3_eq_0)
//{
//	if (!mesh.is_valid())
//		return;
//	auto domain0 = make_domain<VERTEX>(mesh);
//	auto domain1 = make_domain<EDGE>(mesh);
//	auto domain2 = make_domain<FACE>(mesh);
//	auto domain3 = make_domain<VOLUME>(mesh);
//	auto f3 = make_field<scalar_type>(domain3);
//	auto f1a = make_field<scalar_type>(domain1);
//	auto f1b = make_field<scalar_type>(domain1);
//	auto f2 = make_field<scalar_type>(domain2);
//	std::mt19937 gen;
//	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);
//
//	Real m = 0.0;
//
//	f3.clear();
//
//	for (auto s : domain3)
//	{
//		auto a = uniform_dist(gen);
//		f3[s] = a * default_value;
//		m += a * a;
//	}
//	update_ghosts(&f3);
//	m = std::sqrt(m) * abs(default_value);
//
//	LOG_CMD(f2 = grad(f3));
//	LOG_CMD(f1a = curl(f2));
//	LOG_CMD(f1b = curl(grad(f3)));
//
//	size_t count = 0;
//	Real variance_a = 0;
//	Real variance_b = 0;
//	for (auto s : domain1)
//	{
//
////		ASSERT_EQ((f1a[s]), (f1b[s]));
//		variance_a += abs(f1a[s]);
//		variance_b += abs(f1b[s]);
//
//	}
//
//	variance_a /= m;
//	variance_b /= m;
//	ASSERT_LE(std::sqrt(variance_b), error);
//	ASSERT_LE(std::sqrt(variance_a), error);
//}
//
//TEST_P(TestFETL, identity_div_curl_f1_eq0)
//{
//	if (!mesh.is_valid())
//		return;
//	auto domain0 = make_domain<VERTEX>(mesh);
//	auto domain1 = make_domain<EDGE>(mesh);
//	auto domain2 = make_domain<FACE>(mesh);
//	auto domain3 = make_domain<VOLUME>(mesh);
//	auto f1 = make_field<scalar_type>(domain1);
//	auto f2 = make_field<scalar_type>(domain2);
//	auto f0a = make_field<scalar_type>(domain0);
//	auto f0b = make_field<scalar_type>(domain0);
//
//	std::mt19937 gen;
//	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);
//
//	f2.clear();
//
//	Real m = 0.0;
//
//	for (auto s : domain2)
//	{
//		auto a = uniform_dist(gen);
//
//		f2[s] = default_value * uniform_dist(gen);
//
//		m += a * a;
//	}
//	update_ghosts(&f2);
//
//	m = std::sqrt(m) * abs(default_value);
//
//	LOG_CMD(f1 = curl(f2));
//
//	LOG_CMD(f0a = diverge(f1));
//
//	LOG_CMD(f0b = diverge(curl(f2)));
//
//	size_t count = 0;
//	Real variance_a = 0;
//	Real variance_b = 0;
//	for (auto s : domain0)
//	{
//		variance_b += abs(f0b[s] * f0b[s]);
//		variance_a += abs(f0a[s] * f0a[s]);
////		ASSERT_EQ((f0a[s]), (f0b[s]));
//	}
//	variance_a /= m;
//	variance_b /= m;
//	ASSERT_LE(std::sqrt(variance_b), error);
//	ASSERT_LE(std::sqrt(variance_a), error);
//
//}
//
//TEST_P(TestFETL, identity_div_curl_f2_eq0)
//{
//	if (!mesh.is_valid())
//		return;
//	auto domain0 = make_domain<VERTEX>(mesh);
//	auto domain1 = make_domain<EDGE>(mesh);
//	auto domain2 = make_domain<FACE>(mesh);
//	auto domain3 = make_domain<VOLUME>(mesh);
//	auto f1 = make_field<scalar_type>(domain1);
//	auto f2 = make_field<scalar_type>(domain2);
//	auto f3a = make_field<scalar_type>(domain3);
//	auto f3b = make_field<scalar_type>(domain3);
//
//	std::mt19937 gen;
//	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);
//
//	f1.clear();
//
//	Real m = 0.0;
//
//	for (auto s : domain1)
//	{
//		auto a = uniform_dist(gen);
//		f1[s] = default_value * a;
//		m += a * a;
//	}
//	update_ghosts(&f1);
//
//	m = std::sqrt(m) * abs(default_value);
//
//	LOG_CMD(f2 = curl(f1));
//
//	LOG_CMD(f3a = diverge(f2));
//
//	LOG_CMD(f3b = diverge(curl(f1)));
//
//	size_t count = 0;
//
//	Real variance_a = 0;
//	Real variance_b = 0;
//	for (auto s : domain3)
//	{
//
////		ASSERT_DOUBLE_EQ(abs(f3a[s]), abs(f3b[s]));
//		variance_a += abs(f3a[s] * f3a[s]);
//		variance_b += abs(f3b[s] * f3b[s]);
//
//	}
//
//	variance_a /= m;
//	variance_b /= m;
//	ASSERT_LE(std::sqrt(variance_b), error);
//	ASSERT_LE(std::sqrt(variance_a), error);
//
//}

#endif /* CORE_FIELD_FIELD_VECTOR_CALCULUS_TEST_H_ */
