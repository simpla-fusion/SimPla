/**
 * @file field_vector_calculus_test.h
 *
 *  Created on: 2014年10月21日
 *      Author: salmon
 */

#ifndef CORE_FIELD_FIELD_VECTOR_CALCULUS_TEST_H_
#define CORE_FIELD_FIELD_VECTOR_CALCULUS_TEST_H_

#include <gtest/gtest.h>
#include <tuple>

#include "../../utilities/utilities.h"
#include "../../field/field.h"
#include "../structured/structured.h"
#include "../../io/io.h"

using namespace simpla;

typedef CartesianRectMesh mesh_type;

typedef nTuple<Real, 3> coordinates_type;

class FETLTest: public testing::TestWithParam<
		std::tuple<nTuple<Real, 3>, nTuple<Real, 3>, nTuple<size_t, 3>,
				nTuple<Real, 3>> >
{
protected:
	void SetUp()
	{
		LOGGER.set_stdout_visable_level(LOG_VERBOSE);
		std::tie(xmin, xmax, dims, K_real) = GetParam();
		K_imag = 0;
		SetDefaultValue(&one);
		for (int i = 0; i < ndims; ++i)
		{
			if (dims[i] <= 1 || (xmax[i] <= xmin[i]))
			{
				dims[i] = 1;
				K_real[i] = 0.0;
				xmax[i] = xmin[i];
			}
		}

		mesh = std::make_shared<mesh_type>();
		mesh->dimensions(&dims[0]);
		mesh->extents(xmin, xmax);
		mesh->deploy();
		Vec3 dx = mesh->dx();
		error = 10 * std::pow(inner_product(K_real, dx), 2.0);
	}
	void TearDown()
	{
		std::shared_ptr<mesh_type>(nullptr).swap(mesh);
	}
public:
	typedef Real value_type;
	typedef typename mesh_type::scalar_type scalar_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	static constexpr size_t ndims = mesh_type::ndims;
	nTuple<Real, 3> xmin;
	nTuple<Real, 3> xmax;
	nTuple<size_t, 3> dims;
	nTuple<Real, 3> K_real; // @NOTE must   k = n TWOPI, period condition
	nTuple<scalar_type, 3> K_imag;
	value_type one;
	Real error;

	std::shared_ptr<mesh_type> mesh;

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
		*v = std::complex<T>(r, 0);
	}
	template<size_t N, typename T>
	void SetDefaultValue(nTuple<T, N>* v)
	{
		for (int i = 0; i < N; ++i)
		{
			(*v)[i] = i;
		}
	}
	virtual ~FETLTest()
	{
	}
};
TEST_P(FETLTest, grad0)
{

	auto f0 = make_form<VERTEX, value_type>(*mesh);
	auto f1 = make_form<EDGE, value_type>(*mesh);
	auto f1b = make_form<EDGE, value_type>(*mesh);

	f0.clear();
	f1.clear();
	f1b.clear();

	for (auto s : mesh->domain<VERTEX>())
	{
		f0[s] = std::sin(inner_product(K_real, mesh->coordinates(s)));
	};
	f0.sync();

	LOG_CMD(f1 = grad(f0));

	Real m = 0.0;
	Real variance = 0;
	value_type average;
	average *= 0.0;

	Real mean = 0;

	for (auto s : mesh->domain<EDGE>())
	{
		size_t n = mesh->sub_index(s);

		auto x = mesh->coordinates(s);

		value_type expect;

		expect = K_real[n] * std::cos(inner_product(K_real, x))
				+ K_imag[n] * std::sin(inner_product(K_real, x));

//		if (mesh->get_type_as_string() == "Cylindrical"
//				&& n == (mesh_type::ZAxis + 1) % 3)
//		{
//			auto r = mesh->coordinates(s);
//			expect /= r[(mesh_type::ZAxis + 2) % 3];
//		}

		f1b[s] = expect;

//		CHECK(expect) << " " << f1[s] << " " << K_real << " " << K_imag
//				<< std::endl;

		variance += mod((f1[s] - expect) * (f1[s] - expect));

		average += (f1[s] - expect);

		m += mod(f1[s]);

//		if (mod(expect) > EPSILON)
//		{
//			EXPECT_LE(mod(2.0 * (f1[s] - expect) / (f1[s] + expect)), error) << " expect = " << expect
//			        << " actual = " << f1[s] << " x= " << mesh->coordinates(s) << " K= " << K_real << " mesh->K="
//			        << mesh->k_imag;
//		}
//		else
//		{
//			EXPECT_LE(mod(f1[s]), error) << " expect = " << expect << " actual = " << f1[s] << " x= "
//			        << mesh->coordinates(s);
//
//		}

//		if (mod(f1[s]) > epsilon || mod(expect) > epsilon)
//		{
//			ASSERT_GE(error, mod(2.0 * (f1[s] - expect) / (f1[s] + expect)));
//		}

	}
//
//	variance /=mesh->template get_num_of_elements<EDGE>();
//	average /=mesh->template get_num_of_elements<EDGE>();
	EXPECT_LE(std::sqrt(variance), error);
	EXPECT_LE(mod(average), error);

	cd("/grad1/");
	LOGGER << SAVE(f0) << endl;
	LOGGER << SAVE(f1) << endl;
	LOGGER << SAVE(f1b) << endl;

}

TEST_P(FETLTest, grad3)
{
	if (!mesh->is_valid())
		return;

	auto f2 = mesh->make_form<FACE, value_type>();
	auto f2b = mesh->make_form<FACE, value_type>();
	auto f3 = mesh->make_form<VOLUME, value_type>();

	f3.clear();
	f2.clear();
	f2b.clear();

	for (auto s : mesh->template domain<VOLUME>())
	{
		f3[s] = std::sin(inner_product(K_real, mesh->coordinates(s)));
	};

	f3.sync();

	LOG_CMD(f2 = grad(f3));

	Real m = 0.0;
	Real variance = 0;
	value_type average = one * 0.0;

	for (auto s : mesh->domain<FACE>())
	{

		size_t n = mesh->sub_index(s);

		auto x = mesh->coordinates(s);

		value_type expect;
		expect = K_real[n] * std::cos(inner_product(K_real, x))
				+ K_imag[n] * std::sin(inner_product(K_real, x));

		if (mesh->get_type_as_string() == "Cylindrical"
				&& n == (mesh_type::ZAxis + 1) % 3)
		{
			auto r = mesh->coordinates(s);
			expect /= r[(mesh_type::ZAxis + 2) % 3];
		}

		f2b[s] = expect;

		variance += mod((f2[s] - expect) * (f2[s] - expect));

		average += (f2[s] - expect);

//		if (mod(expect) > EPSILON)
//		{
//			EXPECT_LE(mod(2.0 * (f2[s] - expect) / (f2[s] + expect)), error) << " expect = " << expect
//			        << " actual = " << f2[s] << " x= " << mesh->coordinates(s) << " K= " << K_real;
//		}
//		else
//		{
//			EXPECT_LE(mod(f2[s]), error) << " expect = " << expect << " actual = " << f2[s] << " x= "
//			        << mesh->coordinates(s);
//
//		}

	}

	variance /= f2.domain().size();
	average /= f2.domain().size();
	EXPECT_LE(std::sqrt(variance), error) << dims;
	EXPECT_LE(mod(average), error);

	cd("/grad3/");
	LOGGER << SAVE(f3) << endl;
	LOGGER << SAVE(f2) << endl;
	LOGGER << SAVE(f2b) << endl;

}

TEST_P(FETLTest, diverge1)
{
	if (!mesh->is_valid())
		return;

	auto f1 = mesh->make_form<EDGE, value_type>();
	auto f0 = mesh->make_form<VERTEX, value_type>();
	auto f0b = mesh->make_form<VERTEX, value_type>();
	f0.clear();
	f0b.clear();
	f1.clear();

	for (auto s : mesh->domain<EDGE>())
	{
		f1[s] = std::sin(inner_product(K_real, mesh->coordinates(s)));
	};
	f1.sync();
	LOG_CMD(f0 = diverge(f1));

//	f0 = codifferential_derivative(f1);
//	f0 = -f0;

	Real variance = 0;

	value_type average;
	average *= 0;

	for (auto s : mesh->domain<VERTEX>())
	{

		auto x = mesh->coordinates(s);

		Real cos_v = std::cos(inner_product(K_real, x));
		Real sin_v = std::sin(inner_product(K_real, x));

		value_type expect;

		if (mesh->get_type_as_string() == "Cylindrical")
		{

			expect =

			(K_real[(mesh_type::ZAxis + 1) % 3] / x[(mesh_type::ZAxis + 2) % 3]
					+ //  k_theta

					K_real[(mesh_type::ZAxis + 2) % 3] + //  k_r

					K_real[(mesh_type::ZAxis + 3) % 3] //  k_z
			) * cos_v

					+ (K_imag[(mesh_type::ZAxis + 1) % 3]
							/ x[(mesh_type::ZAxis + 2) % 3] + //  k_theta

							K_imag[(mesh_type::ZAxis + 2) % 3] + //  k_r

							K_imag[(mesh_type::ZAxis + 3) % 3] //  k_z
					) * sin_v;

			expect += sin_v / x[(mesh_type::ZAxis + 2) % 3]; //A_r
		}
		else
		{
			expect = (K_real[0] + K_real[1] + K_real[2]) * cos_v
					+ (K_imag[0] + K_imag[1] + K_imag[2]) * sin_v;
		}

		f0b[s] = expect;

		variance += mod((f0[s] - expect) * (f0[s] - expect));

		average += (f0[s] - expect);

//		if (mod(expect) > EPSILON)
//		{
//			EXPECT_LE(mod(2.0 * (f0[s] - expect) / (f0[s] + expect)), error) << " expect = " << expect
//			        << " actual = " << f0[s] << " x= " << mesh->coordinates(s) << " K= " << K_real << " K_i= "
//			        << K_imag;
//		}
//		else
//		{
//			EXPECT_LE(mod(f0[s]), error) << " expect = " << expect << " actual = " << f0[s] << " x= "
//			        << mesh->coordinates(s);
//
//		}
	}

	variance /= f0.domain().size();
	average /= f0.domain().size();

	CHECK(average);

	EXPECT_LE(std::sqrt(variance), error) << dims;
	EXPECT_LE(mod(average), error) << " K= " << K_real << " K_i= " << K_imag

//			<< " mesh->Ki=" << mesh->k_imag

			;

}

TEST_P(FETLTest, diverge2)
{
	if (!mesh->is_valid())
		return;

	auto f2 = mesh->make_form<FACE, value_type>();
	auto f3 = mesh->make_form<VOLUME, value_type>();

	f3.clear();
	f2.clear();

	for (auto s : mesh->domain<FACE>())
	{
		f2[s] = std::sin(inner_product(K_real, mesh->coordinates(s)));
	};
	f2.sync();

	LOG_CMD(f3 = diverge(f2));

	Real variance = 0;
	value_type average;
	average *= 0.0;

	for (auto s : mesh->domain<VOLUME>())
	{
		auto x = mesh->coordinates(s);

		Real cos_v = std::cos(inner_product(K_real, x));
		Real sin_v = std::sin(inner_product(K_real, x));

		value_type expect;

		if (mesh->get_type_as_string() == "Cylindrical")
		{

			expect =

			(K_real[(mesh_type::ZAxis + 1) % 3] / x[(mesh_type::ZAxis + 2) % 3]
					+ //  k_theta

					K_real[(mesh_type::ZAxis + 2) % 3] + //  k_r

					K_real[(mesh_type::ZAxis + 3) % 3] //  k_z
			) * cos_v

					+ (K_imag[(mesh_type::ZAxis + 1) % 3]
							/ x[(mesh_type::ZAxis + 2) % 3] + //  k_theta

							K_imag[(mesh_type::ZAxis + 2) % 3] + //  k_r

							K_imag[(mesh_type::ZAxis + 3) % 3] //  k_z
					) * sin_v;

			expect += std::sin(inner_product(K_real, mesh->coordinates(s)))
					/ x[(mesh_type::ZAxis + 2) % 3]; //A_r
		}
		else
		{
			expect = (K_real[0] + K_real[1] + K_real[2]) * cos_v
					+ (K_imag[0] + K_imag[1] + K_imag[2]) * sin_v;
		}

		variance += mod((f3[s] - expect) * (f3[s] - expect));

		average += (f3[s] - expect);

//		if (mod(f3[s]) > epsilon || mod(expect) > epsilon)
//			ASSERT_LE(mod(2.0 * (f3[s] - expect) / (f3[s] + expect)), error);

	}

	variance /= f3.domain().size();
	average /= f3.domain().size();

	ASSERT_LE(std::sqrt(variance), error);
	ASSERT_LE(mod(average), error);

}

TEST_P(FETLTest, curl1)
{
	if (!mesh->is_valid())
		return;

	auto f1 = mesh->make_form<EDGE, value_type>();
	auto f1b = mesh->make_form<EDGE, value_type>();
	auto f2 = mesh->make_form<FACE, value_type>();
	auto f2b = mesh->make_form<FACE, value_type>();

	f1.clear();
	f1b.clear();
	f2.clear();
	f2b.clear();

	Real m = 0.0;
	Real variance = 0;
	value_type average;
	average *= 0.0;

	for (auto s : mesh->domain<EDGE>())
	{
		f1[s] = std::sin(inner_product(K_real, mesh->coordinates(s)));
	};
	f1.sync();
	LOG_CMD(f2 = curl(f1));

	for (auto s : mesh->domain<FACE>())
	{
		auto n = mesh->sub_index(s);

		auto x = mesh->coordinates(s);

		Real cos_v = std::cos(inner_product(K_real, x));
		Real sin_v = std::sin(inner_product(K_real, x));

		value_type expect;

		if (mesh->get_type_as_string() == "Cylindrical")
		{
			switch (n)
			{
			case (mesh_type::ZAxis + 1) % 3: // theta
				expect = (K_real[(mesh_type::ZAxis + 2) % 3]
						- K_real[(mesh_type::ZAxis + 3) % 3]) * cos_v
						+ (K_imag[(mesh_type::ZAxis + 2) % 3]
								- K_imag[(mesh_type::ZAxis + 3) % 3]) * sin_v;
				break;
			case (mesh_type::ZAxis + 2) % 3: // r
				expect = (K_real[(mesh_type::ZAxis + 3) % 3]
						- K_real[(mesh_type::ZAxis + 1) % 3]
								/ x[(mesh_type::ZAxis + 2) % 3]) * cos_v

						+ (K_imag[(mesh_type::ZAxis + 3) % 3]
								- K_imag[(mesh_type::ZAxis + 1) % 3]
										/ x[(mesh_type::ZAxis + 2) % 3])
								* sin_v;
				break;

			case (mesh_type::ZAxis + 3) % 3: // z
				expect = (K_real[(mesh_type::ZAxis + 1) % 3]
						- K_real[(mesh_type::ZAxis + 2) % 3]) * cos_v
						+ (K_imag[(mesh_type::ZAxis + 1) % 3]
								- K_imag[(mesh_type::ZAxis + 2) % 3]) * sin_v;

				expect -= std::sin(inner_product(K_real, mesh->coordinates(s)))
						/ x[(mesh_type::ZAxis + 2) % 3]; //A_r
				break;

			}

		}
		else
		{
			expect = (K_real[(n + 1) % 3] - K_real[(n + 2) % 3]) * cos_v
					+ (K_imag[(n + 1) % 3] - K_imag[(n + 2) % 3]) * sin_v;
		}

		f2b[s] = expect;

		variance += mod((f2[s] - expect) * (f2[s] - expect));

		average += (f2[s] - expect);

//		if (mod(expect) > epsilon)
//		{
//			EXPECT_LE(mod(2.0 * (vf2[s] - expect) / (vf2[s] + expect)), error) << " expect = " << expect
//			        << " actual = " << vf2[s] << " x= " << mesh->coordinates(s);
//		}
//		else
//		{
//			EXPECT_LE(mod(vf2[s]), error) << " expect = " << expect << " actual = " << vf2[s] << " x= "
//			        << mesh->coordinates(s);
//		}

	}

	variance /= f2.domain().size();
	average /= f2.domain().size();

	ASSERT_LE(std::sqrt(variance), error);
	ASSERT_LE(mod(average), error);

}

TEST_P(FETLTest, curl2)
{
	if (!mesh->is_valid())
		return;

	auto f1 = mesh->make_form<EDGE, value_type>();
	auto vf1b = mesh->make_form<EDGE, value_type>();
	auto f2 = mesh->make_form<FACE, value_type>();
	auto vf2b = mesh->make_form<FACE, value_type>();

	f1.clear();
	vf1b.clear();
	f2.clear();
	vf2b.clear();

	Real m = 0.0;
	Real variance = 0;
	value_type average;
	average *= 0.0;

	for (auto s : mesh->domain<FACE>())
	{
		f2[s] = std::sin(inner_product(K_real, mesh->coordinates(s)));
	};
	f2.sync();
	LOG_CMD(f1 = curl(f2));
//	f1 = codifferential_derivative(f2);
//	f1 = -f1;

	vf1b.clear();

	for (auto s : mesh->domain<EDGE>())
	{

		auto n = mesh->sub_index(s);

		auto x = mesh->coordinates(s);

		Real cos_v = std::cos(inner_product(K_real, x));
		Real sin_v = std::sin(inner_product(K_real, x));

		value_type expect;

		if (mesh->get_type_as_string() == "Cylindrical")
		{
			switch (n)
			{
			case (mesh_type::ZAxis + 1) % 3: // theta
				expect = (K_real[(mesh_type::ZAxis + 2) % 3]
						- K_real[(mesh_type::ZAxis + 3) % 3]) * cos_v
						+ (K_imag[(mesh_type::ZAxis + 2) % 3]
								- K_imag[(mesh_type::ZAxis + 3) % 3]) * sin_v;
				break;
			case (mesh_type::ZAxis + 2) % 3: // r
				expect = (K_real[(mesh_type::ZAxis + 3) % 3]
						- K_real[(mesh_type::ZAxis + 1) % 3]
								/ x[(mesh_type::ZAxis + 2) % 3]) * cos_v

						+ (K_imag[(mesh_type::ZAxis + 3) % 3]
								- K_imag[(mesh_type::ZAxis + 1) % 3]
										/ x[(mesh_type::ZAxis + 2) % 3])
								* sin_v;
				break;

			case (mesh_type::ZAxis + 3) % 3: // z
				expect = (K_real[(mesh_type::ZAxis + 1) % 3]
						- K_real[(mesh_type::ZAxis + 2) % 3]) * cos_v
						+ (K_imag[(mesh_type::ZAxis + 1) % 3]
								- K_imag[(mesh_type::ZAxis + 2) % 3]) * sin_v;

				expect -= std::sin(inner_product(K_real, mesh->coordinates(s)))
						/ x[(mesh_type::ZAxis + 2) % 3]; //A_r
				break;

			}

		}
		else
		{
			expect = (K_real[(n + 1) % 3] - K_real[(n + 2) % 3]) * cos_v
					+ (K_imag[(n + 1) % 3] - K_imag[(n + 2) % 3]) * sin_v;
		}

		vf1b[s] = expect;

		variance += mod((f1[s] - expect) * (f1[s] - expect));

		average += (f1[s] - expect);

//		if (mod(expect) > epsilon)
//		{
//			ASSERT_LE(mod(2.0 * (vf1[s] - expect) / (vf1[s] + expect)), error)<< " expect = "<<expect<<" actual = "<<vf1[s]<< " x= "<<mesh->coordinates(s);
//		}
//		else
//		{
//			ASSERT_LE(mod(vf1[s]), error)<< " expect = "<<expect<<" actual = "<<vf1[s]<< " x= "<<mesh->coordinates(s);
//
//		}

	}

//	GLOBAL_DATA_STREAM.cd("/");
//	LOGGER << SAVE(vf2);
//	LOGGER << SAVE(vf1);
//	LOGGER << SAVE(vf1b);
	variance /= f1.domain().size();
	average /= f1.domain().size();

	ASSERT_LE(std::sqrt(variance), error);
	ASSERT_LE(mod(average), error);

}

TEST_P(FETLTest, identity_curl_grad_f0_eq_0)
{
	if (!mesh->is_valid())
		return;

	auto f0 = mesh->make_form<VERTEX, value_type>();
	auto f1 = mesh->make_form<EDGE, value_type>();
	auto f2a = mesh->make_form<FACE, value_type>();
	auto f2b = mesh->make_form<FACE, value_type>();

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m = 0.0;
	f0.clear();
	for (auto s : mesh->domain<VERTEX>())
	{

		auto a = uniform_dist(gen);
		f0[s] = one * a;
		m += a * a;
	}
	f0.sync();

	m = std::sqrt(m) * mod(one);

	LOG_CMD(f1 = grad(f0));
	LOG_CMD(f2a = curl(f1));
	LOG_CMD(f2b = curl(grad(f0)));

	size_t count = 0;
	Real variance_a = 0;
	Real variance_b = 0;
	for (auto s : mesh->domain<FACE>())
	{

		variance_a += mod(f2a[s]);
		variance_b += mod(f2b[s]);
//		ASSERT_EQ((f2a[s]), (f2b[s]));
	}

	variance_a /= m;
	variance_b /= m;
	ASSERT_LE(std::sqrt(variance_b), error);
	ASSERT_LE(std::sqrt(variance_a), error);

}

TEST_P(FETLTest, identity_curl_grad_f3_eq_0)
{
	if (!mesh->is_valid())
		return;

	auto f3 = mesh->make_form<VOLUME, value_type>();
	auto f1a = mesh->make_form<EDGE, value_type>();
	auto f1b = mesh->make_form<EDGE, value_type>();
	auto f2 = mesh->make_form<FACE, value_type>();
	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m = 0.0;

	f3.clear();

	for (auto s : mesh->domain<VOLUME>())
	{
		auto a = uniform_dist(gen);
		f3[s] = a * one;
		m += a * a;
	}
	f3.sync();
	m = std::sqrt(m) * mod(one);

	LOG_CMD(f2 = grad(f3));
	LOG_CMD(f1a = curl(f2));
	LOG_CMD(f1b = curl(grad(f3)));

	size_t count = 0;
	Real variance_a = 0;
	Real variance_b = 0;
	for (auto s : mesh->domain<EDGE>())
	{

//		ASSERT_EQ((f1a[s]), (f1b[s]));
		variance_a += mod(f1a[s]);
		variance_b += mod(f1b[s]);

	}

	variance_a /= m;
	variance_b /= m;
	ASSERT_LE(std::sqrt(variance_b), error);
	ASSERT_LE(std::sqrt(variance_a), error);
}

TEST_P(FETLTest, identity_div_curl_f1_eq0)
{
	if (!mesh->is_valid())
		return;

	auto f1 = mesh->make_form<EDGE, value_type>();
	auto f2 = mesh->make_form<FACE, value_type>();
	auto f0a = mesh->make_form<VERTEX, value_type>();
	auto f0b = mesh->make_form<VERTEX, value_type>();

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	f2.clear();

	Real m = 0.0;

	for (auto s : mesh->domain<FACE>())
	{
		auto a = uniform_dist(gen);

		f2[s] = one * uniform_dist(gen);

		m += a * a;
	}
	f2.sync();

	m = std::sqrt(m) * mod(one);

	LOG_CMD(f1 = curl(f2));

	LOG_CMD(f0a = diverge(f1));

	LOG_CMD(f0b = diverge(curl(f2)));

	size_t count = 0;
	Real variance_a = 0;
	Real variance_b = 0;
	for (auto s : mesh->domain<VERTEX>())
	{
		variance_b += mod(f0b[s] * f0b[s]);
		variance_a += mod(f0a[s] * f0a[s]);
//		ASSERT_EQ((f0a[s]), (f0b[s]));
	}
	variance_a /= m;
	variance_b /= m;
	ASSERT_LE(std::sqrt(variance_b), error);
	ASSERT_LE(std::sqrt(variance_a), error);

}

TEST_P(FETLTest, identity_div_curl_f2_eq0)
{

	auto f1 = mesh->make_form<EDGE, value_type>();
	auto f2 = mesh->make_form<FACE, value_type>();
	auto f3a = mesh->make_form<VOLUME, value_type>();
	auto f3b = mesh->make_form<VOLUME, value_type>();

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	f1.clear();

	Real m = 0.0;

	for (auto s : mesh->domain<EDGE>())
	{
		auto a = uniform_dist(gen);
		f1[s] = one * a;
		m += a * a;
	}
	f1.sync();

	m = std::sqrt(m) * mod(one);

	LOG_CMD(f2 = curl(f1));

	LOG_CMD(f3a = diverge(f2));

	LOG_CMD(f3b = diverge(curl(f1)));

	size_t count = 0;

	Real variance_a = 0;
	Real variance_b = 0;
	for (auto s : mesh->domain<VOLUME>())
	{

//		ASSERT_DOUBLE_EQ(mod(f3a[s]), mod(f3b[s]));
		variance_a += mod(f3a[s] * f3a[s]);
		variance_b += mod(f3b[s] * f3b[s]);

	}

	variance_a /= m;
	variance_b /= m;
	ASSERT_LE(std::sqrt(variance_b), error);
	ASSERT_LE(std::sqrt(variance_a), error);

}
//
//INSTANTIATE_TEST_CASE_P(FETLTEST, FETLTest,
//
//testing::Combine(testing::Values(
//
//nTuple<Real, 3>( { 0.0, 0.0, 0.0 })
//
//, nTuple<Real, 3>( { -1.0, -2.0, -3.0 })
//
//),
//
//testing::Values(
//
//nTuple<Real, 3>( { 1.0, 2.0, 1.0 }) //
//
//		, nTuple<Real, 3>( { 2.0, 0.0, 0.0 }) //
//		, nTuple<Real, 3>( { 0.0, 2.0, 0.0 }) //
//		, nTuple<Real, 3>( { 0.0, 0.0, 2.0 }) //
//		, nTuple<Real, 3>( { 0.0, 2.0, 2.0 }) //
//		, nTuple<Real, 3>( { 2.0, 0.0, 2.0 }) //
//		, nTuple<Real, 3>( { 2.0, 2.0, 0.0 }) //
//
//		),
//
//testing::Values(
//
//nTuple<size_t, 3>( { 40, 12, 10 }) //
//		, nTuple<size_t, 3>( { 100, 1, 1 }) //
//		, nTuple<size_t, 3>( { 1, 100, 1 }) //
//		, nTuple<size_t, 3>( { 1, 1, 100 }) //
//		, nTuple<size_t, 3>( { 1, 10, 5 }) //
//		, nTuple<size_t, 3>( { 11, 1, 21 }) //
//		, nTuple<size_t, 3>( { 11, 21, 1 }) //
//		),
//
//testing::Values(
//
//nTuple<Real, 3>( { TWOPI, 3 * TWOPI, TWOPI }))
//
//));

#endif /* CORE_FIELD_FIELD_VECTOR_CALCULUS_TEST_H_ */
