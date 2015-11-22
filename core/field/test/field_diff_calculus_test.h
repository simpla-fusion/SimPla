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

#include "../../manifold/manifold_traits.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/primitives.h"

#include "../../gtl/utilities/log.h"

using namespace simpla;

//#define CYLINDRICAL_COORDINATE_SYSTEM 1

#ifdef CYLINDRICAL_COORDINATE_SYSTEM

#include "../../manifold/pre_define/cylindrical.h"

typedef manifold::Cylindrical mesh_type;
typedef geometry::traits::coordinate_system_t<mesh_type> cs;
static constexpr const int RAxis = cs::RAxis;
static constexpr const int ZAxis = cs::ZAxis;
static constexpr const int PhiAxis = cs::PhiAxis;

#else

#include "../../manifold/pre_define/cartesian.h"
#include "../../manifold/mesh/rect_mesh.h"

typedef manifold::Cartesian mesh_type;

#endif

class FETLTest : public testing::TestWithParam<
        std::tuple<std::tuple<nTuple<Real, 3>, nTuple<Real, 3> >, nTuple<size_t, 3>,
                nTuple<Real, 3>>>
{
protected:
    void SetUp()
    {
        logger::set_stdout_level(logger::LOG_VERBOSE);

        std::tie(box, dims, K_real) = GetParam();

        std::tie(xmin, xmax) = box;

        K_imag = 0;

        for (int i = 0; i < ndims; ++i)
        {
            if (dims[i] <= 1)
            {
                K_real[i] = 0;
            }
            else
            {

                K_real[i] /= (xmax[i] - xmin[i]);
            }
        }


        mesh = std::make_shared<mesh_type>();
        mesh->dimensions(dims);
        mesh->box(box);
        mesh->deploy();

        Vec3 dx = mesh->dx();


        point_type xm;
        xm = (std::get<0>(box) + std::get<1>(box)) * 0.5;


#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        error = 2 * power2(K_real[RAxis] * dx[RAxis] + K_real[ZAxis] * dx[ZAxis] + K_real[PhiAxis] * dx[PhiAxis]);
#else
        error = 2 * power2(inner_product(K_real, dx));
#endif
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
    point_type xmin, xmax;
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

    auto f0 = traits::make_field<value_type, VERTEX>(*mesh);
    auto f1 = traits::make_field<value_type, EDGE>(*mesh);
    auto f1b = traits::make_field<value_type, EDGE>(*mesh);

    f0.

            clear();

    f1.

            clear();

    f1b.

            clear();

    for (auto const &s : mesh->template range<VERTEX>())
    {
        f0[s] = std::sin(inner_product(K_real, mesh->point(s)));
    };
    f0.sync();

    LOG_CMD(f1 = grad(f0));

    Real m = 0.0;
    Real variance = 0;
    value_type average;
    average *= 0.0;

    Real mean = 0;

    size_t count = 0;

    for (auto const &s :mesh->template range<EDGE>())
    {
        ++count;
        size_t n = mesh->sub_index(s);

        auto x = mesh->point(s);

        value_type expect;

        expect = K_real[n] * std::cos(inner_product(K_real, x))
//                 + K_imag[n] * std::sin(inner_product(K_real, x))
                ;

#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        if (n == PhiAxis)
        {
            expect /= x[RAxis];
        }
#endif
        f1b[s] = expect;

        variance += mod((f1[s] - expect) * (f1[s] - expect));

        average += (f1[s] - expect);

        m += mod(f1[s]);

    }

    EXPECT_LE(std::sqrt(variance / count), error
    );
    EXPECT_LE(mod(average) / count, error);
#ifndef NDEBUG
    io::cd("/grad1/");
    LOGGER << SAVE(f0) << std::endl;
    LOGGER << SAVE(f1) << std::endl;
    LOGGER << SAVE(f1b) << std::endl;
#endif
}

TEST_P(FETLTest, grad3)
{


    auto f2 = traits::make_field<value_type, FACE>(*mesh);
    auto f2b = traits::make_field<value_type, FACE>(*mesh);
    auto f3 = traits::make_field<value_type, VOLUME>(*mesh);

    f3.clear();

    f2.clear();

    f2b.clear();

    for (auto s :   mesh->template range<VOLUME>())
    {
        f3[s] = std::sin(inner_product(K_real, mesh->point(s)));
    };

    f3.sync();

    LOG_CMD(f2 = grad(f3));

    Real m = 0.0;
    Real variance = 0;
    value_type average = one * 0.0;
    size_t count = 0;
    for (auto s :  mesh->template range<FACE>())
    {
        ++count;

        size_t n = mesh->sub_index(s);

        auto x = mesh->point(s);

        value_type expect;

        expect = K_real[n] * std::cos(inner_product(K_real, x))
                 + K_imag[n] * std::sin(inner_product(K_real, x));


#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        if (n == PhiAxis) { expect /= x[RAxis]; }
#endif
        f2b[s] = expect;

        variance += mod((f2[s] - expect) * (f2[s] - expect));

        average += (f2[s] - expect);


    }

#ifndef NDEBUG
    io::cd("/grad3/");
    LOGGER << SAVE(f3) << std::endl;
    LOGGER << SAVE(f2) << std::endl;
    LOGGER << SAVE(f2b) << std::endl;
#endif

    EXPECT_LE(std::sqrt(variance / count), error);
    EXPECT_LE(mod(average / count), error);


}

TEST_P(FETLTest, diverge1)
{

    auto f1 = traits::make_field<value_type, EDGE>(*mesh);
    auto f0 = traits::make_field<value_type, VERTEX>(*mesh);
    auto f0b = traits::make_field<value_type, VERTEX>(*mesh);

    f0.clear();

    f0b.clear();

    f1.clear();

    nTuple<Real, 3> E = {1, 2, 3};

    for (auto s : mesh->template range<EDGE>())
    {
        f1[s] = E[mesh->sub_index(s)] * std::sin(inner_product(K_real, mesh->point(s)));
    };

    f1.sync();

    LOG_CMD(f0 = diverge(f1));


    Real variance = 0;

    value_type average;
    average *= 0;

    size_t count = 0;

    for (auto s : mesh->range<VERTEX>())
    {
        auto x = mesh->point(s);

        Real cos_v = std::cos(inner_product(K_real, x));
        Real sin_v = std::sin(inner_product(K_real, x));

        value_type expect;

#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        expect = K_real[PhiAxis] * E[PhiAxis] / x[RAxis] * cos_v

                 + E[RAxis] * (K_real[RAxis] * cos_v + sin_v / x[RAxis])

                 + K_real[ZAxis] * E[ZAxis] * cos_v;

#else
        expect = (K_real[0] * E[0] + K_real[1] * E[1] + K_real[2] * E[2]) * cos_v;
#endif
        f0b[s] = expect;


#ifdef CYLINDRICAL_COORDINATE_SYSTEM

        if (dims[RAxis] > 1 && mesh->sub_index(s) == RAxis && mesh->idx_to_boundary(s) <= 1)
        {
            continue;
        }
#endif

        ++count;

        variance += mod((f0[s] - expect) * (f0[s] - expect));

        average += (f0[s] - expect);


    }

    EXPECT_GT(count, 0);

    variance /= count;
    average /= count;

#ifndef NDEBUG
    io::cd("/div1/");
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

    auto f2 = traits::make_field<value_type, FACE>(*mesh);
    auto f3 = traits::make_field<value_type, VOLUME>(*mesh);
    auto f3b = traits::make_field<value_type, VOLUME>(*mesh);
    f3.clear();

    f3b.clear();

    f2.clear();

    for (auto s :  mesh->template range<FACE>())
    {
        f2[s] = std::sin(inner_product(K_real, mesh->point(s)));
    };
    f2.sync();

    LOG_CMD(f3 = diverge(f2));

    Real variance = 0;
    value_type average;
    average *= 0.0;

    size_t count = 0;

    for (auto s :mesh->template range<VOLUME>())
    {

        auto x = mesh->point(s);

        Real cos_v = std::cos(inner_product(K_real, x));
        Real sin_v = std::sin(inner_product(K_real, x));

        value_type expect;

#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        expect =
                (K_real[PhiAxis] / x[RAxis] + K_real[RAxis] + K_real[ZAxis]) * cos_v
                + (K_imag[PhiAxis] / x[RAxis] + K_imag[RAxis] + K_imag[ZAxis]) * sin_v;

        expect += sin_v / x[RAxis];
#else
        expect = (K_real[0] + K_real[1] + K_real[2]) * cos_v
                 + (K_imag[0] + K_imag[1] + K_imag[2]) * sin_v;
#endif


        f3b[s] = expect;

#ifdef CYLINDRICAL_COORDINATE_SYSTEM

        if (dims[mesh->sub_index(s)] > 1 && mesh->idx_to_boundary(s) <= 1) { continue; }
#endif
        ++count;


        variance += mod((f3[s] - expect) * (f3[s] - expect));

        average += (f3[s] - expect);


    }

    variance /= count;
    average /= count;


#ifndef NDEBUG

    io::cd("/div2/");
    LOGGER << SAVE(f2) << std::endl;
    LOGGER << SAVE(f3) << std::endl;
    LOGGER << SAVE(f3b) << std::endl;
#endif

    ASSERT_LE(std::sqrt(variance), error);
    ASSERT_LE(mod(average), error);

}

TEST_P(FETLTest, curl1)
{


    auto f1 = traits::make_field<value_type, EDGE>(*mesh);
    auto f1b = traits::make_field<value_type, EDGE>(*mesh);
    auto f2 = traits::make_field<value_type, FACE>(*mesh);
    auto f2b = traits::make_field<value_type, FACE>(*mesh);

    f1.clear();

    f1b.clear();

    f2.clear();

    f2b.clear();

    Real m = 0.0;
    Real variance = 0;
    value_type average;
    average *= 0.0;


    nTuple<Real, 3> E = {0, 0, 3};

    for (auto s : mesh->range<EDGE>())
    {
        f1[s] = E[mesh->sub_index(s)] * std::sin(inner_product(K_real, mesh->point(s)));
    };


    f1.sync();

    LOG_CMD(f2 = curl(f1));

    size_t count = 0;

    for (auto s :mesh->template range<FACE>())
    {
        auto n = mesh->sub_index(s);

        auto x = mesh->point(s);

        Real cos_v = std::cos(inner_product(K_real, x));
        Real sin_v = std::sin(inner_product(K_real, x));

        value_type expect;

#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        switch (n)
        {
            case RAxis:// r
                expect = (K_real[PhiAxis] * E[ZAxis] / x[RAxis] - K_real[ZAxis] * E[RAxis]) * cos_v;
                break;

            case ZAxis:// z
                expect = K_real[PhiAxis] * E[RAxis] * cos_v / x[RAxis] - K_real[RAxis] * E[PhiAxis] * cos_v -
                         E[PhiAxis] * sin_v / x[RAxis];
                break;

            case PhiAxis: // theta
                expect = (K_real[RAxis] * E[ZAxis] - K_real[ZAxis] * E[RAxis]) * cos_v;
                break;


        }

#else
        expect = (K_real[(n + 1) % 3] * E[(n + 2) % 3] - K_real[(n + 2) % 3] * E[(n + 1) % 3]) * cos_v;

#endif

#ifdef CYLINDRICAL_COORDINATE_SYSTEM

        if (dims[mesh->sub_index(s)] > 1 && mesh->idx_to_boundary(s) <= 1) { continue; }
#endif
        ++count;

        f2b[s] = expect;

        variance += mod((f2[s] - expect) * (f2[s] - expect));

        average += (f2[s] - expect);


    }

#ifndef NDEBUG
    io::cd("/curl1/");
    LOGGER << SAVE(f1) << std::endl;
    LOGGER << SAVE(f2) << std::endl;
    LOGGER << SAVE(f2b) << std::endl;
#endif


    ASSERT_LE(std::sqrt(variance / count), error);
    ASSERT_LE(mod(average / count), error);

}

TEST_P(FETLTest, curl2)
{


    auto f1 = traits::make_field<value_type, EDGE>(*mesh);
    auto f1b = traits::make_field<value_type, EDGE>(*mesh);
    auto f2 = traits::make_field<value_type, FACE>(*mesh);
    auto f2b = traits::make_field<value_type, FACE>(*mesh);

    f1.clear();

    f1b.clear();

    f2.clear();

    f2b.clear();

    Real m = 0.0;
    Real variance = 0;
    value_type average;
    average *= 0.0;

    for (auto s :  mesh->template range<FACE>())
    {
        f2[s] = std::sin(inner_product(K_real, mesh->point(s)));
    };

    f2.sync();

    LOG_CMD(f1 = curl(f2));

    f1b.clear();

    size_t count = 0;

    for (auto s :mesh->template range<EDGE>())
    {

        auto n = mesh->sub_index(s);

        auto x = mesh->point(s);

        value_type expect;


#ifdef CYLINDRICAL_COORDINATE_SYSTEM

        Real cos_v = std::cos(inner_product(K_real, x));
        Real sin_v = std::sin(inner_product(K_real, x));

        switch (n)
        {
            case PhiAxis: // theta
                expect = (K_real[RAxis] - K_real[ZAxis]) * cos_v
                         + (K_imag[RAxis] - K_imag[ZAxis]) * sin_v;
                break;
            case RAxis:// r
                expect = (K_real[ZAxis] - K_real[PhiAxis] / x[RAxis]) * cos_v
                         + (K_imag[ZAxis] - K_imag[PhiAxis] / x[RAxis]) * sin_v;
                break;

            case ZAxis:// z
                expect = (K_real[PhiAxis] - K_real[RAxis]) * cos_v
                         + (K_imag[PhiAxis] - K_imag[RAxis]) * sin_v;


                expect -= sin_v / x[RAxis];
                break;

        }

#	else

        Real cos_v = std::cos(inner_product(K_real, x));

        Real sin_v = std::sin(inner_product(K_real, x));


        expect = (K_real[(n + 1) % 3] - K_real[(n + 2) % 3]) * cos_v
                 + (K_imag[(n + 1) % 3] - K_imag[(n + 2) % 3]) * sin_v;

#endif
        f1b[s] = expect;

#ifdef CYLINDRICAL_COORDINATE_SYSTEM

        if (dims[mesh->sub_index(s)] > 1 && mesh->idx_to_boundary(s) <= 1) { continue; }
#endif
        ++count;

        variance += mod((f1[s] - expect) * (f1[s] - expect));

        average += (f1[s] - expect);


    }
#ifndef NDEBUG

    io::cd("/curl2/");
    LOGGER << SAVE(f2) << std::endl;
    LOGGER << SAVE(f1) << std::endl;
    LOGGER << SAVE(f1b) << std::endl;
#endif


    ASSERT_LE(std::sqrt(variance / count), error);
    ASSERT_LE(mod(average / count), error);

}


TEST_P(FETLTest, identity_curl_grad_f0_eq_0)
{

    auto f0 = traits::make_field<value_type, VERTEX>(*mesh);
    auto f1 = traits::make_field<value_type, EDGE>(*mesh);
    auto f2a = traits::make_field<value_type, FACE>(*mesh);
    auto f2b = traits::make_field<value_type, FACE>(*mesh);

    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    Real m = 0.0;
    f0.clear();

    for (auto s : mesh->template range<VERTEX>())
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
    for (auto s :   mesh->template range<FACE>())
    {

        variance_a += mod(f2a[s]);
        variance_b += mod(f2b[s]);
    }

    variance_a /= m;
    variance_b /= m;
    ASSERT_LE(std::sqrt(variance_b), error);
    ASSERT_LE(std::sqrt(variance_a), error);

}

TEST_P(FETLTest, identity_curl_grad_f3_eq_0
)
{


    auto f3 = traits::make_field<value_type, VOLUME>(*mesh);
    auto f1a = traits::make_field<value_type, EDGE>(*mesh);
    auto f1b = traits::make_field<value_type, EDGE>(*mesh);
    auto f2 = traits::make_field<value_type, FACE>(*mesh);
    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    Real m = 0.0;

    f3.clear();

    for (auto s :mesh->template range<VOLUME>())
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
    for (auto s : mesh->template range<EDGE>())
    {
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


    auto f1 = traits::make_field<value_type, EDGE>(*mesh);
    auto f2 = traits::make_field<value_type, FACE>(*mesh);
    auto f0a = traits::make_field<value_type, VERTEX>(*mesh);
    auto f0b = traits::make_field<value_type, VERTEX>(*mesh);

    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    f2.clear();

    Real m = 0.0;

    for (auto s :  mesh->template range<FACE>())
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


    for (auto s :  mesh->template range<VERTEX>())
    {

#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        if (dims[mesh->sub_index(s)] > 1 && mesh->idx_to_boundary(s) <= 1) { continue; }
#endif
        ++count;

        variance_b += mod(f0b[s] * f0b[s]);
        variance_a += mod(f0a[s] * f0a[s]);
//		ASSERT_EQ((f0a[s]), (f0b[s]));
    }

    ASSERT_LE(std::sqrt(variance_b / count), error);
    ASSERT_LE(std::sqrt(variance_a / count), error);

}

TEST_P(FETLTest, identity_div_curl_f2_eq0
)
{

    auto f1 = traits::make_field<value_type, EDGE>(*mesh);
    auto f2 = traits::make_field<value_type, FACE>(*mesh);
    auto f3a = traits::make_field<value_type, VOLUME>(*mesh);
    auto f3b = traits::make_field<value_type, VOLUME>(*mesh);

    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    f1.clear();

    Real m = 0.0;

    for (auto s : mesh->range<EDGE>())
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

    for (auto s :mesh->template range<VOLUME>())
    {

#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        if (dims[mesh->sub_index(s)] > 1 && mesh->idx_to_boundary(s) <= 1) { continue; }
#endif
        ++count;

        variance_a += mod(f3a[s] * f3a[s]);
        variance_b += mod(f3b[s] * f3b[s]);

    }


    EXPECT_LE(std::sqrt(variance_b / count), error);

    ASSERT_LE(std::sqrt(variance_a / count), error);

}

#endif /* CORE_FIELD_VECTOR_CALCULUS_TEST_H_ */
