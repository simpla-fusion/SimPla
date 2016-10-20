/**
 * @file field_vector_calculus_test.h
 *
 *  Created on: 2014-10-21
 *      Author: salmon
 */

#ifndef CORE_FIELD_VECTOR_CALCULUS_TEST_H_
#define CORE_FIELD_VECTOR_CALCULUS_TEST_H_

#include <simpla/SIMPLA_config.h>

#include <stddef.h>
#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <string>

#include <gtest/gtest.h>
#include <simpla/toolbox/IO.h>
#include <simpla/toolbox/nTuple.h>
#include <simpla/toolbox/Log.h>

#include <simpla/mesh/Mesh.h>
#include <simpla/physics/Field.h>
#include <simpla/physics/FieldExpression.h>
#include <simpla/physics/FieldTraits.h>
#include <simpla/manifold/Calculus.h>
#include <simpla/manifold/pre_define/PreDefine.h>


using namespace simpla;
using namespace simpla::mesh;
using namespace simpla::calculus;
//#define CYLINDRICAL_COORDINATE_SYSTEM 1

#ifdef CYLINDRICAL_COORDINATE_SYSTEM
typedef manifold::CylindricalManifold mesh_type;
typedef geometry::traits::coordinate_system_t<mesh_type> cs;
static constexpr const int RAxis = cs::RAxis;
static constexpr const int ZAxis = cs::ZAxis;
static constexpr const int PhiAxis = cs::PhiAxis;

#else

typedef manifold::CartesianManifold mesh_type;

#endif

class FETLTest : public testing::TestWithParam<
        std::tuple<std::tuple<nTuple<Real, 3>, nTuple<Real, 3> >, nTuple<size_t, 3>,
                nTuple<Real, 3>>>
{
protected:
    void SetUp()
    {
        logger::set_stdout_level(logger::LOG_VERBOSE);
        m = std::make_shared<mesh_type>();
        std::tie(box, dims, K_real) = GetParam();
        std::tie(xmin, xmax) = box;
        m->ghost_width(2, 2, 2);
        m->dimensions(dims);

#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        m->m_ghost_width_(index_tuple({2, 0, 0}));
#endif
        m->box(box);
        m->deploy();
        Vec3 dx;
        dx = m->dx();
        point_type xm;
        xm = (std::get<0>(box) + std::get<1>(box)) * 0.5;
        K_imag = 0;
        for (int i = 0; i < ndims; ++i)
        {
            if (dims[i] <= 1) { K_real[i] = 0; }
            else { K_real[i] /= (xmax[i] - xmin[i]); }
        }


//#ifdef CYLINDRICAL_COORDINATE_SYSTEM
//        //        K_real[PhiAxis]/=xm[RAxis];
//                error = 2 * power2(K_real[RAxis] * dx[RAxis] + K_real[ZAxis] * dx[ZAxis] + K_real[PhiAxis] * dx[PhiAxis] );
//#else
        error = 2 * power2(inner_product(K_real, dx));
//#endif
        one = 1;
    }

    void TearDown() {}

public:
    typedef Real value_type;

    static constexpr size_t ndims = mesh_type::ndims;

    box_type box;
    point_type xmin, xmax;
    size_tuple dims;
    vector_type K_real; // @NOTE must   k = n TWOPI, period condition
    vector_type K_imag;
    value_type one;
    Real error;


    std::shared_ptr<mesh_type> m;


    Real q(point_type const &x) const
    {
//#ifdef CYLINDRICAL_COORDINATE_SYSTEM
//        res = K_real[RAxis] *  x[RAxis] + K_real[ZAxis] *  x[ZAxis] + K_real[PhiAxis] *  x[PhiAxis]  ;
//#else
        Real res = K_real[0] * x[0] + K_real[1] * x[1] + K_real[2] * x[2];// inner_product(K_real, x);
//#endif
        return res;
    }

    virtual ~FETLTest()
    {
    }

    template<typename TV, size_t IEntityType>
    field_t<TV, mesh_type, IEntityType> make_field() { return field_t<TV, mesh_type, IEntityType>(m); };
};

TEST_P(FETLTest, grad0)
{
    typedef MeshEntityIdCoder M;

    auto f0 = make_field<value_type, VERTEX>();
    auto f1 = make_field<value_type, EDGE>();
    auto f1b = make_field<value_type, EDGE>();

    f0.clear();
    f1.clear();
    f1b.clear();


    m->range(VERTEX, SP_ES_ALL).foreach([&](mesh::MeshEntityId s) { f0[s] = std::sin(q(m->point(s))); });

    LOG_CMD(f1 = grad(f0));

    Real variance = 0;
    value_type average;
    average *= 0.0;

    m->range(EDGE, SP_ES_OWNED).foreach(
            [&](mesh::MeshEntityId s)
            {
                int n = M::sub_index(s);
                auto x = m->point(s);
                value_type expect;
                expect = K_real[n] * std::cos(q(x))
//                 + K_imag[n] * std::sin(inner_product(K_real, x))
                        ;

//#ifdef CYLINDRICAL_COORDINATE_SYSTEM
//                if(n==PhiAxis) { expect /=x[RAxis]; }
//#endif
                f1b[s] = expect;
                variance += mod((f1[s] - expect) * (f1[s] - expect));
                average += (f1[s] - expect);
                EXPECT_LE(abs(f1[s] - expect), error)
                                    << expect << "," << f1[s] << "[" << (s.x >> 1) << "," << (s.y >> 1) <<
                                    "," << (s.z >> 1) << "]" << std::endl;
            });

    EXPECT_LE(std::sqrt(variance / m->range(EDGE).size()), error);
    EXPECT_LE(mod(average) / m->range(EDGE).size(), error);

//#ifndef NDEBUG
//    io::cd("/grad1/");
//    LOGGER << SAVE(f0) << std::endl;
//    LOGGER << SAVE(f1) << std::endl;
//    LOGGER << SAVE(f1b) << std::endl;
//#endif
}

TEST_P(FETLTest, grad3)
{
    typedef MeshEntityIdCoder M;

    auto f2 = make_field<value_type, FACE>();
    auto f2b = make_field<value_type, FACE>();
    auto f3 = make_field<value_type, VOLUME>();

    f3.clear();
    f2.clear();
    f2b.clear();

    m->range(VOLUME, SP_ES_ALL).foreach([&](mesh::MeshEntityId s) { f3[s] = std::sin(q(m->point(s))); });

    LOG_CMD(f2 = grad(f3));

    Real variance = 0;
    value_type average = one * 0;

    m->range(FACE, SP_ES_OWNED).foreach(
            [&](mesh::MeshEntityId s)
            {
                int n = M::sub_index(s);
                auto x = m->point(s);
                value_type expect;
                expect = K_real[n] * std::cos(q(x));

//#ifdef CYLINDRICAL_COORDINATE_SYSTEM
//                if(n==PhiAxis) { expect /=x[RAxis]; }
//#endif

                f2b[s] = expect;
                variance += mod((f2[s] - expect) * (f2[s] - expect));
                average += (f2[s] - expect);

                EXPECT_LE(abs(f2[s] - expect), error)
                                    << expect << "," << f2[s] << "[" << (s.x >> 1) << "," << (s.y >> 1) <<
                                    "," << (s.z >> 1) << "]" << std::endl;
            });

//#ifndef NDEBUG
//    io::cd("/grad3/");
//    LOGGER << SAVE(f3) << std::endl;
//    LOGGER << SAVE(f2) << std::endl;
//    LOGGER << SAVE(f2b) << std::endl;
//#endif

    EXPECT_LE(std::sqrt(variance / m->range(FACE).size()), error);
    EXPECT_LE(mod(average / m->range(FACE).size()), error);
}

TEST_P(FETLTest, diverge1)
{
    typedef MeshEntityIdCoder M;

    auto f1 = make_field<value_type, EDGE>();
    auto f0 = make_field<value_type, VERTEX>();
    auto f0b = make_field<value_type, VERTEX>();

    f0.clear();
    f0b.clear();
    f1.clear();

    nTuple<Real, 3> E{1, 1, 1};

    m->range(EDGE, SP_ES_ALL).foreach(
            [&](mesh::MeshEntityId s)
            {
                f1[s] = E[M::sub_index(s)] * std::sin(q(m->point(s)));
            });

    LOG_CMD(f0 = diverge(f1));
    Real variance = 0;
    value_type average;
    average = 0;
    m->range(VERTEX, SP_ES_OWNED).foreach(
            [&](mesh::MeshEntityId s)
            {
                auto x = m->point(s);

                Real cos_v = std::cos(q(x));
                Real sin_v = std::sin(q(x));
                value_type expect;

//#ifdef CYLINDRICAL_COORDINATE_SYSTEM
//                expect = K_real[PhiAxis] * E[PhiAxis] / x[RAxis] * cos_v
//                         + E[RAxis] * (K_real[RAxis] * cos_v + sin_v / x[RAxis])
//                         + K_real[ZAxis] * E[ZAxis] * cos_v;
//#else
                expect = (K_real[0] * E[0] + K_real[1] * E[1] + K_real[2] * E[2]) * cos_v;
//#endif
                f0b[s] = expect;
                variance += mod((f0[s] - expect) * (f0[s] - expect));
                average += (f0[s] - expect);

                EXPECT_LE(abs(f0[s] - expect), error)
                                    << expect << "," << f0[s] << "[" << (s.x >> 1) << "," << (s.y >> 1) <<
                                    "," << (s.z >> 1) << "]" << std::endl;
            });

    EXPECT_LE(std::sqrt(variance /= m->range(VERTEX).size()), error);
    EXPECT_LE(mod(average /= m->range(VERTEX).size()), error);
}

TEST_P(FETLTest, diverge2)
{
    typedef MeshEntityIdCoder M;
    auto f2 = make_field<value_type, FACE>();
    auto f3 = make_field<value_type, VOLUME>();
    auto f3b = make_field<value_type, VOLUME>();
    f3.clear();
    f3b.clear();
    f2.clear();

    m->range(FACE, SP_ES_ALL).foreach([&](mesh::MeshEntityId s) { f2[s] = std::sin(q(m->point(s))); });
    LOG_CMD(f3 = diverge(f2));

    Real variance = 0;
    value_type average;
    average *= 0.0;

    m->range(VOLUME, SP_ES_OWNED).foreach(
            [&](mesh::MeshEntityId s)
            {

                auto x = m->point(s);

                Real cos_v = std::cos(q(x));
                Real sin_v = std::sin(q(x));

                value_type expect;

//#ifdef CYLINDRICAL_COORDINATE_SYSTEM
//                expect =
//                        (K_real[PhiAxis] / x[RAxis] + K_real[RAxis] + K_real[ZAxis]) * cos_v
//                        + (K_imag[PhiAxis] / x[RAxis] + K_imag[RAxis] + K_imag[ZAxis]) * sin_v;
//
//                expect += sin_v / x[RAxis];
//#else
                expect = (K_real[0] + K_real[1] + K_real[2]) * cos_v
                         + (K_imag[0] + K_imag[1] + K_imag[2]) * sin_v;
//#endif
                f3b[s] = expect;
                variance += mod((f3[s] - expect) * (f3[s] - expect));
                average += (f3[s] - expect);


                EXPECT_LE(abs(f3[s] - expect), error)
                                    << expect << "," << f3[s] << "[" << (s.x >> 1) << "," << (s.y >> 1) <<
                                    "," << (s.z >> 1) << "]" << std::endl;
            }
    );
    variance /= m->range(VOLUME).size();
    average /= m->range(VOLUME).size();


//#ifndef NDEBUG
//
//    io::cd("/div2/");
//    LOGGER << SAVE(f2) << std::endl;
//    LOGGER << SAVE(f3) << std::endl;
//    LOGGER << SAVE(f3b) << std::endl;
//#endif

    ASSERT_LE(std::sqrt(variance), error);
    ASSERT_LE(mod(average), error);

}

TEST_P(FETLTest, curl1)
{
    typedef MeshEntityIdCoder M;

    auto f1 = make_field<value_type, EDGE>();
    auto f1b = make_field<value_type, EDGE>();
    auto f2 = make_field<value_type, FACE>();
    auto f2b = make_field<value_type, FACE>();

    f1.clear();

    f1b.clear();

    f2.clear();

    f2b.clear();

    Real variance = 0;
    value_type average;
    average *= 0.0;


    nTuple<Real, 3> E = {1, 1, 1};

    m->range(EDGE, SP_ES_ALL).foreach(
            [&](mesh::MeshEntityId s)
            {
                f1[s] = E[M::sub_index(s)] * std::sin(q(m->point(s)));
            });

    LOG_CMD(f2 = curl(f1));

    m->range(FACE).foreach(
            [&](mesh::MeshEntityId s)
            {
                auto n = M::sub_index(s);

                auto x = m->point(s);

                Real cos_v = std::cos(q(x));
                Real sin_v = std::sin(q(x));
                value_type expect;

//#ifdef CYLINDRICAL_COORDINATE_SYSTEM
//                //        switch (n)
//                //        {
//                //            case   RAxis:// r
//                //                expect = (K_real[PhiAxis] * E[ZAxis] / x[RAxis] - K_real[ZAxis] * E[RAxis]) * cos_v;
//                //                break;
//                //
//                //            case   ZAxis:// z
//                //                expect = (K_real[PhiAxis] * E[RAxis]  / x[RAxis] - K_real[RAxis] * E[PhiAxis]) * cos_v
//                //                      -   E[PhiAxis] * sin_v / x[RAxis];
//                //                break;
//                //
//                //            case  PhiAxis: // theta
//                //                expect = (K_real[RAxis] * E[ZAxis] - K_real[ZAxis] * E[RAxis]) * cos_v;
//                //                break;
//                //
//                //
//                //        }
//
//
//                switch (n)
//                {
//                    case  PhiAxis: // theta
//                        expect = (K_real[RAxis]*E[ZAxis] - K_real[ZAxis]*E[RAxis]) * cos_v;
//                         break;
//                    case  RAxis:// r
//                        expect = (K_real[ZAxis]*E[PhiAxis] - K_real[PhiAxis]*E[ZAxis] / x[RAxis]) * cos_v
//                         ;
//                        break;
//
//                    case ZAxis:// z
//                        expect = (K_real[PhiAxis]*E[RAxis] / x[RAxis]  - K_real[RAxis]*E[PhiAxis] ) * cos_v ;
//
//                        expect -= sin_v*E[PhiAxis]  / x[RAxis];
//                        break;
//
//                }
//#else
                expect = (K_real[(n + 1) % 3] * E[(n + 2) % 3] - K_real[(n + 2) % 3] * E[(n + 1) % 3]) * cos_v;
//#endif


                f2b[s] = expect;
                variance += mod((f2[s] - expect) * (f2[s] - expect));
                average += (f2[s] - expect);


            }
    );
//#ifndef NDEBUG
//    io::cd("/curl1/");
//    LOGGER << SAVE(f1) << std::endl;
//    LOGGER << SAVE(f2) << std::endl;
//    LOGGER << SAVE(f2b) << std::endl;
//#endif


    ASSERT_LE(std::sqrt(variance / m->range(FACE).size()), error);
    ASSERT_LE(mod(average / m->range(FACE).size()), error);

}

TEST_P(FETLTest, curl2)
{

    typedef MeshEntityIdCoder M;
    auto f1 = make_field<value_type, EDGE>();
    auto f1b = make_field<value_type, EDGE>();
    auto f2 = make_field<value_type, FACE>();
    auto f2b = make_field<value_type, FACE>();

    f1.clear();
    f1b.clear();
    f2.clear();
    f2b.clear();

    Real variance = 0;
    value_type average;
    average = 0.0;
    nTuple<Real, 3> E = {1, 2, 3};

    m->range(FACE, SP_ES_ALL).foreach(
            [&](mesh::MeshEntityId s) { f2[s] = E[M::sub_index(s)] * std::sin(q(m->point(s))); });


    LOG_CMD(f1 = curl(f2));
    f1b.clear();

    m->range(EDGE).foreach(
            [&](mesh::MeshEntityId s)
            {

                auto n = M::sub_index(s);
                auto x = m->point(s);
                value_type expect;

                Real cos_v = std::cos(q(x));
                Real sin_v = std::sin(q(x));

//#ifdef CYLINDRICAL_COORDINATE_SYSTEM
//                switch (n)
//               {
//                   case  PhiAxis: // theta
//                       expect = (K_real[RAxis]*E[ZAxis] - K_real[ZAxis]*E[RAxis]) * cos_v;
//                        break;
//                   case  RAxis:// r
//                       expect = (K_real[ZAxis]*E[PhiAxis] - K_real[PhiAxis]*E[ZAxis] / x[RAxis]) * cos_v
//                        ;
//                       break;
//
//                   case ZAxis:// z
//                       expect = (K_real[PhiAxis]*E[RAxis]/ x[RAxis]  - K_real[RAxis]*E[PhiAxis] ) * cos_v ;
//                       expect -= sin_v*E[PhiAxis]  / x[RAxis];
//                       break;
//
//               }
//#	else
                expect = (K_real[(n + 1) % 3] * E[(n + 2) % 3] - K_real[(n + 2) % 3] * E[(n + 1) % 3]) * cos_v;
//#endif
                f1b[s] = expect;
                variance += mod((f1[s] - expect) * (f1[s] - expect));
                average += (f1[s] - expect);

                EXPECT_LE(abs(f1[s] - expect), error)
                                    << expect << "," << f1[s] << "[" << (s.x >> 1) << "," << (s.y >> 1) <<
                                    "," << (s.z >> 1) << "]" << std::endl;
            }
    );
//#ifndef NDEBUG
//
//    //io::cd("/curl2/");
//    LOGGER << SAVE(f2) << std::endl;
//    LOGGER << SAVE(f1) << std::endl;
//    LOGGER << SAVE(f1b) << std::endl;
//#endif


    ASSERT_LE(std::sqrt(variance / m->range(EDGE).size()), error);
    ASSERT_LE(mod(average / m->range(EDGE).size()), error);

}


TEST_P(FETLTest, identity_curl_grad_f0_eq_0)
{

    auto f0 = make_field<value_type, VERTEX>();
    auto f1 = make_field<value_type, EDGE>();
    auto f2a = make_field<value_type, FACE>();
    auto f2b = make_field<value_type, FACE>();

    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    Real mean = 0.0;
    f0.clear();
    m->range(VERTEX, SP_ES_ALL).foreach(
            [&](mesh::MeshEntityId s)
            {
                auto a = uniform_dist(gen);
                f0[s] = one * a;
                mean += a * a;
            });

    mean = std::sqrt(mean) * mod(one);

    LOG_CMD(f1 = grad(f0));
    LOG_CMD(f2a = curl(f1));
    LOG_CMD(f2b = curl(grad(f0)));

    Real variance_a = 0;
    Real variance_b = 0;

    m->range(FACE, SP_ES_OWNED).foreach(
            [&](mesh::MeshEntityId s)
            {
                variance_a += mod(f2a[s]);
                variance_b += mod(f2b[s]);
            }
    );
    variance_a /= mean;
    variance_b /= mean;
    ASSERT_LE(std::sqrt(variance_b), error);
    ASSERT_LE(std::sqrt(variance_a), error);

}

TEST_P(FETLTest, identity_curl_grad_f3_eq_0)
{
    auto f3 = make_field<value_type, VOLUME>();
    auto f1a = make_field<value_type, EDGE>();
    auto f1b = make_field<value_type, EDGE>();
    auto f2 = make_field<value_type, FACE>();
    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    Real mean = 0.0;

    f3.clear();
    m->range(VOLUME, SP_ES_ALL).foreach(
            [&](mesh::MeshEntityId s)
            {
                auto a = uniform_dist(gen);
                f3[s] = a * one;
                mean += a * a;
            });

    mean = std::sqrt(mean) * mod(one);

    LOG_CMD(f2 = grad(f3));
    LOG_CMD(f1a = curl(f2));
    LOG_CMD(f1b = curl(grad(f3)));

    size_t count = 0;
    Real variance_a = 0;
    Real variance_b = 0;
    m->range(EDGE, SP_ES_OWNED).foreach(
            [&](mesh::MeshEntityId s)
            {
                variance_a += mod(f1a[s]);
                variance_b += mod(f1b[s]);
            });

    variance_a /= mean;
    variance_b /= mean;
    ASSERT_LE(std::sqrt(variance_b), error);
    ASSERT_LE(std::sqrt(variance_a), error);
}

TEST_P(FETLTest, identity_div_curl_f1_eq0)
{


    auto f1 = make_field<value_type, EDGE>();
    auto f2 = make_field<value_type, FACE>();
    auto f0a = make_field<value_type, VERTEX>();
    auto f0b = make_field<value_type, VERTEX>();

    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    f2.clear();
    Real mean = 0.0;
    m->range(FACE, SP_ES_ALL).foreach(
            [&](mesh::MeshEntityId s)
            {
                auto a = uniform_dist(gen);
                f2[s] = one * uniform_dist(gen);
                mean += a * a;
            });

    mean = std::sqrt(mean) * mod(one);

    LOG_CMD(f1 = curl(f2));
    LOG_CMD(f0a = diverge(f1));
    LOG_CMD(f0b = diverge(curl(f2)));


    Real variance_a = 0;
    Real variance_b = 0;

    m->range(VERTEX, SP_ES_OWNED).foreach(
            [&](mesh::MeshEntityId s)
            {
                variance_b += mod(f0b[s] * f0b[s]);
                variance_a += mod(f0a[s] * f0a[s]);
            });

    ASSERT_LE(std::sqrt(variance_b / m->range(VERTEX, SP_ES_OWNED).size()), error);
    ASSERT_LE(std::sqrt(variance_a / m->range(VERTEX, SP_ES_OWNED).size()), error);

}

TEST_P(FETLTest, identity_div_curl_f2_eq0)
{

    auto f1 = make_field<value_type, EDGE>();
    auto f2 = make_field<value_type, FACE>();
    auto f3a = make_field<value_type, VOLUME>();
    auto f3b = make_field<value_type, VOLUME>();

    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    f1.clear();

    Real mean = 0.0;
    size_type count = 0;
    m->range(EDGE, SP_ES_ALL).foreach(
            [&](mesh::MeshEntityId s)
            {
                auto a = uniform_dist(gen);
                f1[s] = one * a;
                ++count;
                mean += a * a;
            });

    mean = std::sqrt(mean) * mod(one);

    LOG_CMD(f2 = curl(f1));
    LOG_CMD(f3a = diverge(f2));
    LOG_CMD(f3b = diverge(curl(f1)));


    Real variance_a = 0;
    Real variance_b = 0;
    m->range(VOLUME, SP_ES_OWNED).foreach(
            [&](mesh::MeshEntityId s)
            {
                variance_a += mod(f3a[s] * f3a[s]);
                variance_b += mod(f3b[s] * f3b[s]);
            });


    EXPECT_LE(std::sqrt(variance_b / m->range(VOLUME, SP_ES_OWNED).size()), error);
    ASSERT_LE(std::sqrt(variance_a / m->range(VOLUME, SP_ES_OWNED).size()), error);

}

#endif /* CORE_FIELD_VECTOR_CALCULUS_TEST_H_ */
