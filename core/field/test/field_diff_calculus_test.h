/**
 * @file field_vector_calculus_test.h
 *
 *  Created on: 2014-10-21
 *      Author: salmon
 */

#ifndef CORE_FIELD_VECTOR_CALCULUS_TEST_H_
#define CORE_FIELD_VECTOR_CALCULUS_TEST_H_

#include <stddef.h>
#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <string>

#include <gtest/gtest.h>
#include "../../io/io.h"

#include "../../field/field_dense.h"
#include "../../field/field_expression.h"
#include "../../field/field_traits.h"

#include "../../manifold/domain.h"
#include "../../manifold/manifold_traits.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/primitives.h"

#include "../../gtl/utilities/log.h"


using namespace simpla;

#ifdef CYLINDRICAL_COORDINATE_SYSTEM

#include "../../manifold/pre_define/cylindrical.h"

typedef manifold::Cylindrical mesh_type;

#else

#	include "../../manifold/pre_define/cartesian.h"

typedef manifold::Cartesian<3> mesh_type;

#endif

class FETLTest : public testing::TestWithParam<
        std::tuple<std::tuple<nTuple<Real, 3>, nTuple<Real, 3> >, nTuple<size_t, 3>,
                nTuple<Real, 3>>>
{
protected:
    void SetUp()
    {
        LOGGER.set_stdout_visable_level(logger::LOG_VERBOSE);

        std::tie(box, dims, K_real) = GetParam();

        point_type xmin, xmax;

        std::tie(xmin, xmax) = box;

        K_imag = 0;

        for (int i = 0; i < ndims; ++i)
        {
            if (dims[i] <= 1) { K_real[i] = 0; }

            K_real[i] /= (xmax[i] - xmin[i]);

            K_imag = 0;
        }


        mesh = std::make_shared<mesh_type>();
        mesh->dimensions(dims);
        mesh->box(box);
        mesh->deploy();

        Vec3 dx = mesh->dx();

        point_type xm;
        xm = (std::get<0>(box) + std::get<1>(box)) * 0.5;

        error = 20 * std::pow(mesh->inner_product(K_real, dx, xm), 2.0);

        one = 1;
    }

    void TearDown()
    {
        std::shared_ptr<mesh_type>(nullptr).swap(mesh);
    }

public:
    typedef Real value_type;
    typedef typename mesh_type::scalar_type scalar_type;
    typedef typename mesh_type::point_type point_type;

    static constexpr size_t ndims = mesh_type::ndims;
    std::tuple<nTuple<double, 3>, nTuple<double, 3>> box;

    nTuple<size_t, 3> dims;
    nTuple<Real, 3> K_real; // @NOTE must   k = n TWOPI, period condition
    nTuple<scalar_type, 3> K_imag;
    value_type one;
    Real error;

    std::shared_ptr<mesh_type> mesh;


    virtual ~FETLTest()
    {
    }
};

TEST_P(FETLTest, grad0)
{

    auto f0 = traits::make_field<VERTEX, value_type>(*mesh);
    auto f1 = traits::make_field<EDGE, value_type>(*mesh);
    auto f1b = traits::make_field<EDGE, value_type>(*mesh);

    f0.clear();
    f1.clear();
    f1b.clear();

    for (auto const &s : traits::make_domain<VERTEX>(*mesh))
    {
        f0[s] = std::sin(mesh->inner_product(K_real, mesh->point(s), mesh->point(s)));
    };
    f0.sync();

    LOG_CMD(f1 = grad(f0));

    Real m = 0.0;
    Real variance = 0;
    value_type average;
    average *= 0.0;

    Real mean = 0;

    size_t count = 0;

    for (auto const &s : traits::make_domain<EDGE>(*mesh))
    {
        ++count;
        size_t n = mesh->sub_index(s);

        auto x = mesh->point(s);

        value_type expect;

        expect = K_real[n] * std::cos(mesh->inner_product(K_real, x, x))
//                 + K_imag[n] * std::sin(mesh->inner_product(K_real, x,x))
                ;

#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        if (n == ( traits::ZAxis<mesh_type>::value + 1) % 3)
        {
            auto r = mesh->point(s);
            expect /= r[( traits::ZAxis<mesh_type>::value + 2) % 3];
        }
#endif
        f1b[s] = expect;

        variance += mod((f1[s] - expect) * (f1[s] - expect));

        average += (f1[s] - expect);

        m += mod(f1[s]);
        //		if (mod(expect) > EPSILON)
//		{
//			EXPECT_LE(mod(2.0 * (f1[s] - expect) / (f1[s] + expect)), error) << " expect = " << expect
//			        << " actual = " << f1[s] << " x= " << geometry->point(s) << " K= " << K_real << " geometry->K="
//			        << geometry->k_imag;
//		}
//		else
//		{
//			EXPECT_LE(mod(f1[s]), error) << " expect = " << expect << " actual = " << f1[s] << " x= "
//			        << geometry->point(s);
//		}
//		if (mod(f1[s]) > epsilon || mod(expect) > epsilon)
//		{
//			ASSERT_GE(error, mod(2.0 * (f1[s] - expect) / (f1[s] + expect)));
//		}

    }

    EXPECT_LE(std::sqrt(variance / count), error);
    EXPECT_LE(mod(average) / count, error);
#ifndef NDEBUG
    cd("/grad1/");
    LOGGER << SAVE(f0) << std::endl;
    LOGGER << SAVE(f1) << std::endl;
    LOGGER << SAVE(f1b) << std::endl;
#endif
}

TEST_P(FETLTest, grad3)
{


    if (!mesh->is_valid()) { return; }

    auto f2 = traits::make_field<FACE, value_type>(*mesh);
    auto f2b = traits::make_field<FACE, value_type>(*mesh);
    auto f3 = traits::make_field<VOLUME, value_type>(*mesh);

    f3.clear();
    f2.clear();
    f2b.clear();

    for (auto s : traits::make_domain<VOLUME>(*mesh))
    {
        f3[s] = std::sin(mesh->inner_product(K_real, mesh->point(s), mesh->point(s)));
    };

    f3.sync();

    LOG_CMD(f2 = grad(f3));

    Real m = 0.0;
    Real variance = 0;
    value_type average = one * 0.0;
    size_t count = 0;
    for (auto s : traits::make_domain<FACE>(*mesh))
    {
        ++count;

        size_t n = mesh->sub_index(s);

        auto x = mesh->point(s);

        value_type expect;
        expect = K_real[n] * std::cos(mesh->inner_product(K_real, x, x))

//                 + K_imag[n] * std::sin(mesh->inner_product(K_real, x,x))
                ;
#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        if (n == ( traits::ZAxis<mesh_type>::value + 1) % 3)
        {
            auto r = mesh->point(s);
            expect /= r[( traits::ZAxis<mesh_type>::value + 2) % 3];
        }
#endif
        f2b[s] = expect;

        variance += mod((f2[s] - expect) * (f2[s] - expect));

        average += (f2[s] - expect);

//		if (mod(expect) > EPSILON)
//		{
//			EXPECT_LE(mod(2.0 * (f2[s] - expect) / (f2[s] + expect)), error) << " expect = " << expect
//			        << " actual = " << f2[s] << " x= " << geometry->point(s) << " K= " << K_real;
//		}
//		else
//		{
//			EXPECT_LE(mod(f2[s]), error) << " expect = " << expect << " actual = " << f2[s] << " x= "
//			        << geometry->point(s);
//
//		}

    }

#ifndef NDEBUG
    cd("/grad3/");
    LOGGER << SAVE(f3) << std::endl;
    LOGGER << SAVE(f2) << std::endl;
    LOGGER << SAVE(f2b) << std::endl;
#endif

    EXPECT_LE(std::sqrt(variance / count), error);
    EXPECT_LE(mod(average / count), error);


}

TEST_P(FETLTest, diverge1)
{

    auto f1 = traits::make_field<EDGE, value_type>(*mesh);
    auto f0 = traits::make_field<VERTEX, value_type>(*mesh);
    auto f0b = traits::make_field<VERTEX, value_type>(*mesh);

    f0.clear();
    f0b.clear();
    f1.clear();

    for (auto s : traits::make_domain<EDGE>(*mesh))
    {
        f1[s] = std::sin(mesh->inner_product(K_real, mesh->point(s), mesh->point(s)));
    };
    f1.sync();
    LOG_CMD(f0 = diverge(f1));

//	f0 = codifferential_derivative(f1);
//	f0 = -f0;

    Real variance = 0;

    value_type average;
    average *= 0;

    size_t count = 0;

    for (auto s : traits::make_domain<VERTEX>(*mesh))
    {
        ++count;

        auto x = mesh->point(s);

        Real cos_v = std::cos(mesh->inner_product(K_real, x, x));
        Real sin_v = std::sin(mesh->inner_product(K_real, x, x));

        value_type expect;

#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        expect =

        (K_real[( traits::ZAxis<mesh_type>::value + 1) % 3]
                / x[( traits::ZAxis<mesh_type>::value + 2) % 3] + //  k_theta

                K_real[( traits::ZAxis<mesh_type>::value + 2) % 3] +//  k_r

                K_real[( traits::ZAxis<mesh_type>::value + 3) % 3]//  k_z
        ) * cos_v

        + (K_imag[( traits::ZAxis<mesh_type>::value + 1) % 3]
                / x[( traits::ZAxis<mesh_type>::value + 2) % 3] +//  k_theta

                K_imag[( traits::ZAxis<mesh_type>::value + 2) % 3] +//  k_r

                K_imag[( traits::ZAxis<mesh_type>::value + 3) % 3]//  k_z
        ) * sin_v;

        expect += sin_v / x[( traits::ZAxis<mesh_type>::value + 2) % 3];//A_r
#else

        expect = (K_real[0] + K_real[1] + K_real[2]) * cos_v

                 + (K_imag[0] + K_imag[1] + K_imag[2]) * sin_v;
#endif
        f0b[s] = expect;

        variance += mod((f0[s] - expect) * (f0[s] - expect));

        average += (f0[s] - expect);

//		if (mod(expect) > EPSILON)
//		{
//			EXPECT_LE(mod(2.0 * (f0[s] - expect) / (f0[s] + expect)), error) << " expect = " << expect
//			        << " actual = " << f0[s] << " x= " << geometry->point(s) << " K= " << K_real << " K_i= "
//			        << K_imag;
//		}
//		else
//		{
//			EXPECT_LE(mod(f0[s]), error) << " expect = " << expect << " actual = " << f0[s] << " x= "
//			        << geometry->point(s);
//
//		}
    }

    variance /= count;
    average /= count;

#ifndef NDEBUG
    cd("/div1/");
    LOGGER << SAVE(f1) << std::endl;
    LOGGER << SAVE(f0) << std::endl;
    LOGGER << SAVE(f0b) << std::endl;
#endif

    EXPECT_LE(std::sqrt(variance), error) << dims;
    EXPECT_LE(mod(average), error) << " K= " << K_real << " K_i= " << K_imag

//			<< " geometry->Ki=" << geometry->k_imag

            ;

}

TEST_P(FETLTest, diverge2)
{
    if (!mesh->is_valid())
    {
        return;
    }

    auto f2 = traits::make_field<FACE, value_type>(*mesh);
    auto f3 = traits::make_field<VOLUME, value_type>(*mesh);

    f3.clear();
    f2.clear();

    for (auto s : traits::make_domain<FACE>(*mesh))
    {
        f2[s] = std::sin(mesh->inner_product(K_real, mesh->point(s), mesh->point(s)));
    };
    f2.sync();

    LOG_CMD(f3 = diverge(f2));

    Real variance = 0;
    value_type average;
    average *= 0.0;

    for (auto s : traits::make_domain<VOLUME>(*mesh))
    {
        auto x = mesh->point(s);

        Real cos_v = std::cos(mesh->inner_product(K_real, x, x));
        Real sin_v = std::sin(mesh->inner_product(K_real, x, x));

        value_type expect;

#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        expect =

        (K_real[( traits::ZAxis<mesh_type>::value + 1) % 3]
                / x[( traits::ZAxis<mesh_type>::value + 2) % 3] + //  k_theta

                K_real[( traits::ZAxis<mesh_type>::value + 2) % 3] +//  k_r

                K_real[( traits::ZAxis<mesh_type>::value + 3) % 3]//  k_z
        ) * cos_v

        + (K_imag[( traits::ZAxis<mesh_type>::value + 1) % 3]
                / x[( traits::ZAxis<mesh_type>::value + 2) % 3] +//  k_theta

                K_imag[( traits::ZAxis<mesh_type>::value + 2) % 3] +//  k_r

                K_imag[( traits::ZAxis<mesh_type>::value + 3) % 3]//  k_z
        ) * sin_v;

        expect += std::sin(mesh->inner_product(K_real,  mesh->point(s),  mesh->point(s)))
        / x[( traits::ZAxis<mesh_type>::value + 2) % 3];//A_r

#	else
        expect = (K_real[0] + K_real[1] + K_real[2]) * cos_v
                 + (K_imag[0] + K_imag[1] + K_imag[2]) * sin_v;
#endif
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
    {
        return;
    }

    auto f1 = traits::make_field<EDGE, value_type>(*mesh);
    auto f1b = traits::make_field<EDGE, value_type>(*mesh);
    auto f2 = traits::make_field<FACE, value_type>(*mesh);
    auto f2b = traits::make_field<FACE, value_type>(*mesh);

    f1.clear();
    f1b.clear();
    f2.clear();
    f2b.clear();

    Real m = 0.0;
    Real variance = 0;
    value_type average;
    average *= 0.0;

    for (auto s : traits::make_domain<EDGE>(*mesh))
    {
        f1[s] = std::sin(mesh->inner_product(K_real, mesh->point(s), mesh->point(s)));
    };
    f1.sync();
    LOG_CMD(f2 = curl(f1));

    for (auto s : traits::make_domain<FACE>(*mesh))
    {
        auto n = mesh->sub_index(s);

        auto x = mesh->point(s);

        Real cos_v = std::cos(mesh->inner_product(K_real, x, x));
        Real sin_v = std::sin(mesh->inner_product(K_real, x, x));

        value_type expect;

#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        switch (n)
        {
            case ( traits::ZAxis<mesh_type>::value + 1) % 3: // theta
            expect = (K_real[( traits::ZAxis<mesh_type>::value + 2) % 3]
                    - K_real[( traits::ZAxis<mesh_type>::value + 3) % 3]) * cos_v
            + (K_imag[( traits::ZAxis<mesh_type>::value + 2) % 3]
                    - K_imag[( traits::ZAxis<mesh_type>::value + 3) % 3])
            * sin_v;
            break;
            case ( traits::ZAxis<mesh_type>::value + 2) % 3:// r
            expect = (K_real[( traits::ZAxis<mesh_type>::value + 3) % 3]
                    - K_real[( traits::ZAxis<mesh_type>::value + 1) % 3]
                    / x[( traits::ZAxis<mesh_type>::value + 2) % 3])
            * cos_v

            + (K_imag[( traits::ZAxis<mesh_type>::value + 3) % 3]
                    - K_imag[( traits::ZAxis<mesh_type>::value + 1) % 3]
                    / x[( traits::ZAxis<mesh_type>::value + 2)
                    % 3]) * sin_v;
            break;

            case ( traits::ZAxis<mesh_type>::value + 3) % 3:// z
            expect = (K_real[( traits::ZAxis<mesh_type>::value + 1) % 3]
                    - K_real[( traits::ZAxis<mesh_type>::value + 2) % 3]) * cos_v
            + (K_imag[( traits::ZAxis<mesh_type>::value + 1) % 3]
                    - K_imag[( traits::ZAxis<mesh_type>::value + 2) % 3])
            * sin_v;

            expect -= std::sin(mesh->inner_product(K_real,  mesh->point(s),  mesh->point(s)))
            / x[( traits::ZAxis<mesh_type>::value + 2) % 3];//A_r
            break;

        }

#else

        expect = (K_real[(n + 1) % 3] - K_real[(n + 2) % 3]) * cos_v
                 + (K_imag[(n + 1) % 3] - K_imag[(n + 2) % 3]) * sin_v;

#endif
        f2b[s] = expect;

        variance += mod((f2[s] - expect) * (f2[s] - expect));

        average += (f2[s] - expect);

//		if (mod(expect) > epsilon)
//		{
//			EXPECT_LE(mod(2.0 * (vf2[s] - expect) / (vf2[s] + expect)), error) << " expect = " << expect
//			        << " actual = " << vf2[s] << " x= " <<  mesh->point(s);
//		}
//		else
//		{
//			EXPECT_LE(mod(vf2[s]), error) << " expect = " << expect << " actual = " << vf2[s] << " x= "
//			        <<  mesh->point(s);
//		}

    }

#ifndef NDEBUG
    cd("/curl1/");
    LOGGER << SAVE(f1) << std::endl;
    LOGGER << SAVE(f2) << std::endl;
    LOGGER << SAVE(f2b) << std::endl;
#endif
    variance /= f2.domain().size();
    average /= f2.domain().size();


    ASSERT_LE(std::sqrt(variance), error);
    ASSERT_LE(mod(average), error);

}

TEST_P(FETLTest, curl2)
{
    if (!mesh->is_valid())
    {
        return;
    }

    auto f1 = traits::make_field<EDGE, value_type>(*mesh);
    auto f1b = traits::make_field<EDGE, value_type>(*mesh);
    auto f2 = traits::make_field<FACE, value_type>(*mesh);
    auto f2b = traits::make_field<FACE, value_type>(*mesh);

    f1.clear();
    f1b.clear();
    f2.clear();
    f2b.clear();

    Real m = 0.0;
    Real variance = 0;
    value_type average;
    average *= 0.0;

    for (auto s : traits::make_domain<FACE>(*mesh))
    {
        f2[s] = std::sin(mesh->inner_product(K_real, mesh->point(s), mesh->point(s)));
    };

    f2.sync();

    LOG_CMD(f1 = curl(f2));

    f1b.clear();

    for (auto s : traits::make_domain<EDGE>(*mesh))
    {

        auto n = mesh->sub_index(s);

        auto x = mesh->point(s);

        value_type expect;


#ifdef CYLINDRICAL_COORDINATE_SYSTEM

        Real cos_v = std::cos(mesh->inner_product(K_real, x,x));
        Real sin_v = std::sin(mesh->inner_product(K_real, x,x));

        switch (n)
        {
            case ( traits::ZAxis<mesh_type>::value + 1) % 3: // theta
            expect = (K_real[( traits::ZAxis<mesh_type>::value + 2) % 3]
                    - K_real[( traits::ZAxis<mesh_type>::value + 3) % 3]) * cos_v
            + (K_imag[( traits::ZAxis<mesh_type>::value + 2) % 3]
                    - K_imag[( traits::ZAxis<mesh_type>::value + 3) % 3])
            * sin_v;
            break;
            case ( traits::ZAxis<mesh_type>::value + 2) % 3:// r
            expect = (K_real[( traits::ZAxis<mesh_type>::value + 3) % 3]
                    - K_real[( traits::ZAxis<mesh_type>::value + 1) % 3]
                    / x[( traits::ZAxis<mesh_type>::value + 2) % 3])
            * cos_v

            + (K_imag[( traits::ZAxis<mesh_type>::value + 3) % 3]
                    - K_imag[( traits::ZAxis<mesh_type>::value + 1) % 3]
                    / x[( traits::ZAxis<mesh_type>::value + 2)
                    % 3]) * sin_v;
            break;

            case ( traits::ZAxis<mesh_type>::value + 3) % 3:// z
            expect = (K_real[( traits::ZAxis<mesh_type>::value + 1) % 3]
                    - K_real[( traits::ZAxis<mesh_type>::value + 2) % 3]) * cos_v
            + (K_imag[( traits::ZAxis<mesh_type>::value + 1) % 3]
                    - K_imag[( traits::ZAxis<mesh_type>::value + 2) % 3])
            * sin_v;

            expect -= std::sin(mesh->inner_product(K_real,  mesh->point(s),  mesh->point(s)))
            / x[( traits::ZAxis<mesh_type>::value + 2) % 3];//A_r
            break;

        }

#	else

        Real cos_v = std::cos(mesh->inner_product(K_real, x, x));

        Real sin_v = std::sin(mesh->inner_product(K_real, x, x));


        expect = (K_real[(n + 1) % 3] - K_real[(n + 2) % 3]) * cos_v
                 + (K_imag[(n + 1) % 3] - K_imag[(n + 2) % 3]) * sin_v;

#endif
        f1b[s] = expect;

        variance += mod((f1[s] - expect) * (f1[s] - expect));

        average += (f1[s] - expect);

//		if (mod(expect) > epsilon)
//		{
//			ASSERT_LE(mod(2.0 * (vf1[s] - expect) / (vf1[s] + expect)), error)<< " expect = "<<expect<<" actual = "<<vf1[s]<< " x= "<< mesh->point(s);
//		}
//		else
//		{
//			ASSERT_LE(mod(vf1[s]), error)<< " expect = "<<expect<<" actual = "<<vf1[s]<< " x= "<< mesh->point(s);
//
//		}

    }
#ifndef NDEBUG

    cd("/curl2/");
    LOGGER << SAVE(f2) << std::endl;
    LOGGER << SAVE(f1) << std::endl;
    LOGGER << SAVE(f1b) << std::endl;
#endif
    variance /= f1.domain().size();
    average /= f1.domain().size();

    ASSERT_LE(std::sqrt(variance), error);
    ASSERT_LE(mod(average), error);

}


TEST_P(FETLTest, identity_curl_grad_f0_eq_0)
{

    auto f0 = traits::make_field<VERTEX, value_type>(*mesh);
    auto f1 = traits::make_field<EDGE, value_type>(*mesh);
    auto f2a = traits::make_field<FACE, value_type>(*mesh);
    auto f2b = traits::make_field<FACE, value_type>(*mesh);

    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    Real m = 0.0;
    f0.clear();
    for (auto s : traits::make_domain<VERTEX>(*mesh))
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
    for (auto s : traits::make_domain<FACE>(*mesh))
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
    {
        return;
    }

    auto f3 = traits::make_field<VOLUME, value_type>(*mesh);
    auto f1a = traits::make_field<EDGE, value_type>(*mesh);
    auto f1b = traits::make_field<EDGE, value_type>(*mesh);
    auto f2 = traits::make_field<FACE, value_type>(*mesh);
    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    Real m = 0.0;

    f3.clear();

    for (auto s : traits::make_domain<VOLUME>(*mesh))
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
    for (auto s : traits::make_domain<EDGE>(*mesh))
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


    auto f1 = traits::make_field<EDGE, value_type>(*mesh);
    auto f2 = traits::make_field<FACE, value_type>(*mesh);
    auto f0a = traits::make_field<VERTEX, value_type>(*mesh);
    auto f0b = traits::make_field<VERTEX, value_type>(*mesh);

    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    f2.clear();

    Real m = 0.0;

    for (auto s : traits::make_domain<FACE>(*mesh))
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
    for (auto s : traits::make_domain<VERTEX>(*mesh))
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

    auto f1 = traits::make_field<EDGE, value_type>(*mesh);
    auto f2 = traits::make_field<FACE, value_type>(*mesh);
    auto f3a = traits::make_field<VOLUME, value_type>(*mesh);
    auto f3b = traits::make_field<VOLUME, value_type>(*mesh);

    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    f1.clear();

    Real m = 0.0;

    for (auto s : traits::make_domain<EDGE>(*mesh))
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
    for (auto s : traits::make_domain<VOLUME>(*mesh))
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

#endif /* CORE_FIELD_VECTOR_CALCULUS_TEST_H_ */
