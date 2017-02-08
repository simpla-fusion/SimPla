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
#include <simpla/predefine/CalculusPolicy.h>

#include <simpla/mesh/EntityId.h>
#include "simpla/algebra/all.h"

using namespace simpla;
using namespace simpla::algebra;
//#define CYLINDRICAL_COORDINATE_SYSTEM 1

//#ifdef CYLINDRICAL_COORDINATE_SYSTEM
// typedef manifold::CylindricalManifold manifold_type;
// typedef geometry::traits::coordinate_system_t<manifold_type> cs;
// static constexpr const int RAxis = cs::RAxis;
// static constexpr const int ZAxis = cs::ZAxis;
// static constexpr const int PhiAxis = cs::PhiAxis;
//
//#else

#include <simpla/predefine/CartesianGeometry.h>

typedef mesh::CartesianGeometry mesh_type;

//#endif

class FETLTest
    : public testing::TestWithParam<std::tuple<std::tuple<nTuple<Real, 3>, nTuple<Real, 3>>,
                                               nTuple<index_type, 3>, nTuple<Real, 3>>> {
   protected:
    void SetUp() {
        logger::set_stdout_level(logger::LOG_VERBOSE);

        std::tie(box, dims, K_real) = GetParam();
        std::tie(xmin, xmax) = box;
        size_type gw[3] = {2, 2, 2};
        index_type lo[3] = {0, 0, 0};
        index_type hi[3] = {dims[0], dims[1], dims[2]};

        {
            Real dx[3] = {1, 1, 1};
            Real x0[3] = {0, 0, 0};
            m_p = std::make_shared<mesh_type>(lo, hi, dx, x0);
        }

        //#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        //        m->m_ghost_width_(index_tuple({2, 0, 0}));
        //#endif
        //        m->box(box);
        m_p->Initialize();
        Vec3 dx;
        dx = m_p->dx();
        point_type xm;
        xm = (std::get<0>(box) + std::get<1>(box)) * 0.5;
        K_imag = 0;
        for (int i = 0; i < ndims; ++i) {
            if (dims[i] <= 1) {
                K_real[i] = 0;
            } else {
                K_real[i] /= (xmax[i] - xmin[i]);
            }
        }

        //#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        //        //        K_real[PhiAxis]/=xm[RAxis];
        //                error = 2 * power2(K_real[RAxis] * dx[RAxis] + K_real[ZAxis] * dx[ZAxis] +
        //                K_real[PhiAxis] * dx[PhiAxis] );
        //#else
        //        error = 2 * power2(inner_product(K_real, dx));
        //#endif
        one = 1;

        m = m_p.get();
    }

    void TearDown() {}

   public:
    typedef Real value_type;

    static constexpr size_t ndims = mesh_type::NDIMS;

    box_type box;
    point_type xmin, xmax;
    index_tuple dims;
    vector_type K_real;  // @NOTE must   k = n TWOPI, period condition
    vector_type K_imag;
    value_type one;
    Real error;

    std::shared_ptr<mesh_type> m_p;
    mesh_type* m;
    typedef typename mesh::mesh_traits<mesh_type>::entity_id entity_id;
    Real q(point_type const& x) const {
        //#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        //        res = K_real[RAxis] *  x[RAxis] + K_real[ZAxis] *  x[ZAxis] + K_real[PhiAxis] *
        //        x[PhiAxis]  ;
        //#else
        Real res =
            K_real[0] * x[0] + K_real[1] * x[1] + K_real[2] * x[2];  // inner_product(K_real, x);
                                                                     //#endif
        return res;
    }

    virtual ~FETLTest() {}

    template <int IFORM>
    using field_type = Field<mesh_type, value_type, IFORM, 1>;
};

TEST_P(FETLTest, grad0) {
    typedef mesh::MeshEntityIdCoder M;
    field_type<VERTEX> f0{m};
    field_type<EDGE> f1(m);
    field_type<EDGE> f1b(m);

    f0.clear();
    f1.clear();
    f1b.clear();

    f0.assign([&](point_type const& x) { return std::sin(q(x)); });
    //    m->Range(VERTEX, mesh::SP_ES_ALL).foreach();

    f1 = grad(f0);

    Real variance = 0;
    value_type average = 0.0;

    m_p->range(mesh::SP_ES_OWNED, EDGE).foreach ([&](entity_id const& s) {
        int n = M::sub_index(s);
        auto x = m_p->point(s);
        value_type expect;
        expect = K_real[n] * std::cos(q(x))
            //                 + K_imag[n] * std::sin(inner_product(K_real, x))
            ;

        //#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        //                if(n==PhiAxis) { expect /=x[RAxis]; }
        //#endif
        f1b[s] = expect;
        variance += abs((f1[s] - expect) * (f1[s] - expect));
        average += (f1[s] - expect);
        EXPECT_LE(abs(f1[s] - expect), error) << expect << "," << f1[s] << "[" << (s.x >> 1) << ","
                                              << (s.y >> 1) << "," << (s.z >> 1) << "]"
                                              << std::endl;
    });

    EXPECT_LE(std::sqrt(variance / m_p->range(mesh::SP_ES_NOT_SHARED, EDGE).size()), error);
    EXPECT_LE(abs(average) / m_p->range(mesh::SP_ES_NOT_SHARED, EDGE).size(), error);

    //#ifndef NDEBUG
    //    io::cd("/grad1/");
    //    LOGGER << SAVE(f0) << std::endl;
    //    LOGGER << SAVE(f1) << std::endl;
    //    LOGGER << SAVE(f1b) << std::endl;
    //#endif
}

TEST_P(FETLTest, grad3) {
    typedef mesh::MeshEntityIdCoder M;

    field_type<FACE> f2(m);
    field_type<FACE> f2b(m);
    field_type<VOLUME> f3(m);

    f3.clear();
    f2.clear();
    f2b.clear();

    f3.assign([&](entity_id const& s) { return std::sin(q(m_p->point(s))); });

    LOG_CMD(f2 = grad(f3));

    Real variance = 0;
    value_type average = one * 0;

    m_p->range(mesh::SP_ES_OWNED, FACE).foreach ([&](entity_id const& s) {
        int n = M::sub_index(s);
        auto x = m_p->point(s);
        value_type expect;
        expect = K_real[n] * std::cos(q(x));

        //#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        //                if(n==PhiAxis) { expect /=x[RAxis]; }
        //#endif

        f2b[s] = expect;
        variance += abs((f2[s] - expect) * (f2[s] - expect));
        average += (f2[s] - expect);

        EXPECT_LE(abs(f2[s] - expect), error) << expect << "," << f2[s] << "[" << (s.x >> 1) << ","
                                              << (s.y >> 1) << "," << (s.z >> 1) << "]"
                                              << std::endl;
    });

    //#ifndef NDEBUG
    //    io::cd("/grad3/");
    //    LOGGER << SAVE(f3) << std::endl;
    //    LOGGER << SAVE(f2) << std::endl;
    //    LOGGER << SAVE(f2b) << std::endl;
    //#endif

    EXPECT_LE(std::sqrt(variance / m_p->range(mesh::SP_ES_NOT_SHARED, FACE).size()), error);
    EXPECT_LE(abs(average / m_p->range(mesh::SP_ES_NOT_SHARED, FACE).size()), error);
}

TEST_P(FETLTest, diverge1) {
    typedef mesh::MeshEntityIdCoder M;

    field_type<EDGE> f1(m);
    field_type<VERTEX> f0(m);
    field_type<VERTEX> f0b(m);

    f0.clear();
    f0b.clear();
    f1.clear();

    nTuple<Real, 3> E{1, 1, 1};

    f1.assign([&](entity_id const& s) { return E[M::sub_index(s)] * std::sin(q(m_p->point(s))); });

    LOG_CMD(f0 = diverge(f1));
    Real variance = 0;
    value_type average;
    average = 0;
    m_p->range(mesh::SP_ES_OWNED, VERTEX).foreach ([&](entity_id const& s) {
        auto x = m_p->point(s);

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
        variance += abs((f0[s] - expect) * (f0[s] - expect));
        average += (f0[s] - expect);

        EXPECT_LE(abs(f0[s] - expect), error) << expect << "," << f0[s] << "[" << (s.x >> 1) << ","
                                              << (s.y >> 1) << "," << (s.z >> 1) << "]"
                                              << std::endl;
    });

    EXPECT_LE(std::sqrt(variance /= m_p->range(mesh::SP_ES_NOT_SHARED, VERTEX).size()), error);
    EXPECT_LE(abs(average /= m_p->range(mesh::SP_ES_NOT_SHARED, VERTEX).size()), error);
}

TEST_P(FETLTest, diverge2) {
    typedef mesh::MeshEntityIdCoder M;
    field_type<FACE> f2(m);
    field_type<VOLUME> f3(m);
    field_type<VOLUME> f3b(m);
    f3.clear();
    f3b.clear();
    f2.clear();

    f2.assign([&](entity_id const& s) { return std::sin(q(m_p->point(s))); });

    LOG_CMD(f3 = diverge(f2));

    Real variance = 0;
    value_type average = 0.0;

    m_p->range(mesh::SP_ES_OWNED, VOLUME).foreach ([&](entity_id const& s) {

        auto x = m_p->point(s);

        Real cos_v = std::cos(q(x));
        Real sin_v = std::sin(q(x));

        value_type expect;

        //#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        //                expect =
        //                        (K_real[PhiAxis] / x[RAxis] + K_real[RAxis] + K_real[ZAxis]) *
        //                        cos_v
        //                        + (K_imag[PhiAxis] / x[RAxis] + K_imag[RAxis] + K_imag[ZAxis]) *
        //                        sin_v;
        //
        //                expect += sin_v / x[RAxis];
        //#else
        expect = (K_real[0] + K_real[1] + K_real[2]) * cos_v +
                 (K_imag[0] + K_imag[1] + K_imag[2]) * sin_v;
        //#endif
        f3b[s] = expect;
        variance += abs((f3[s] - expect) * (f3[s] - expect));
        average += (f3[s] - expect);

        EXPECT_LE(abs(f3[s] - expect), error) << expect << "," << f3[s] << "[" << (s.x >> 1) << ","
                                              << (s.y >> 1) << "," << (s.z >> 1) << "]"
                                              << std::endl;
    });
    variance /= m_p->range(mesh::SP_ES_NOT_SHARED, VOLUME).size();
    average /= m_p->range(mesh::SP_ES_NOT_SHARED, VOLUME).size();

    //#ifndef NDEBUG
    //
    //    io::cd("/div2/");
    //    LOGGER << SAVE(f2) << std::endl;
    //    LOGGER << SAVE(f3) << std::endl;
    //    LOGGER << SAVE(f3b) << std::endl;
    //#endif

    ASSERT_LE(std::sqrt(variance), error);
    ASSERT_LE(abs(average), error);
}

TEST_P(FETLTest, curl1) {
    typedef mesh::MeshEntityIdCoder M;

    field_type<EDGE> f1(m);
    field_type<EDGE> f1b(m);
    field_type<FACE> f2(m);
    field_type<FACE> f2b(m);

    f1.clear();

    f1b.clear();

    f2.clear();

    f2b.clear();

    Real variance = 0;
    value_type average;
    average *= 0.0;

    nTuple<Real, 3> E = {1, 1, 1};

    f1.assign([&](entity_id const& s) { return E[M::sub_index(s)] * std::sin(q(m_p->point(s))); });

    LOG_CMD(f2 = curl(f1));

    m_p->range(mesh::SP_ES_NOT_SHARED, FACE).foreach ([&](entity_id const& s) {
        auto n = M::sub_index(s);

        auto x = m_p->point(s);

        Real cos_v = std::cos(q(x));
        Real sin_v = std::sin(q(x));
        value_type expect;

        //#ifdef CYLINDRICAL_COORDINATE_SYSTEM
        //                //        switch (n)
        //                //        {
        //                //            case   RAxis:// r
        //                //                expect = (K_real[PhiAxis] * E[ZAxis] / x[RAxis] -
        //                K_real[ZAxis] * E[RAxis]) * cos_v;
        //                //                break;
        //                //
        //                //            case   ZAxis:// z
        //                //                expect = (K_real[PhiAxis] * E[RAxis]  / x[RAxis] -
        //                K_real[RAxis] * E[PhiAxis]) * cos_v
        //                //                      -   E[PhiAxis] * sin_v / x[RAxis];
        //                //                break;
        //                //
        //                //            case  PhiAxis: // theta
        //                //                expect = (K_real[RAxis] * E[ZAxis] - K_real[ZAxis] *
        //                E[RAxis]) * cos_v;
        //                //                break;
        //                //
        //                //
        //                //        }
        //
        //
        //                switch (n)
        //                {
        //                    case  PhiAxis: // theta
        //                        expect = (K_real[RAxis]*E[ZAxis] - K_real[ZAxis]*E[RAxis]) *
        //                        cos_v;
        //                         break;
        //                    case  RAxis:// r
        //                        expect = (K_real[ZAxis]*E[PhiAxis] - K_real[PhiAxis]*E[ZAxis] /
        //                        x[RAxis]) * cos_v
        //                         ;
        //                        break;
        //
        //                    case ZAxis:// z
        //                        expect = (K_real[PhiAxis]*E[RAxis] / x[RAxis]  -
        //                        K_real[RAxis]*E[PhiAxis] ) * cos_v ;
        //
        //                        expect -= sin_v*E[PhiAxis]  / x[RAxis];
        //                        break;
        //
        //                }
        //#else
        expect =
            (K_real[(n + 1) % 3] * E[(n + 2) % 3] - K_real[(n + 2) % 3] * E[(n + 1) % 3]) * cos_v;
        //#endif

        f2b[s] = expect;
        variance += abs((f2[s] - expect) * (f2[s] - expect));
        average += (f2[s] - expect);

    });
    //#ifndef NDEBUG
    //    io::cd("/curl1/");
    //    LOGGER << SAVE(f1) << std::endl;
    //    LOGGER << SAVE(f2) << std::endl;
    //    LOGGER << SAVE(f2b) << std::endl;
    //#endif

    ASSERT_LE(std::sqrt(variance / m_p->range(mesh::SP_ES_NOT_SHARED, FACE).size()), error);
    ASSERT_LE(abs(average / m_p->range(mesh::SP_ES_NOT_SHARED, FACE).size()), error);
}

TEST_P(FETLTest, curl2) {
    typedef mesh::MeshEntityIdCoder M;
    field_type<EDGE> f1(m);
    field_type<EDGE> f1b(m);
    field_type<FACE> f2(m);
    field_type<FACE> f2b(m);

    f1.clear();
    f1b.clear();
    f2.clear();
    f2b.clear();

    Real variance = 0;
    value_type average;
    average = 0.0;
    nTuple<Real, 3> E = {1, 2, 3};

    f2.assign([&](entity_id const& s) { return E[M::sub_index(s)] * std::sin(q(m_p->point(s))); });

    LOG_CMD(f1 = curl(f2));
    f1b.clear();

    m_p->range(mesh::SP_ES_NOT_SHARED, EDGE).foreach ([&](entity_id const& s) {

        auto n = M::sub_index(s);
        auto x = m_p->point(s);
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
        //                       expect = (K_real[ZAxis]*E[PhiAxis] - K_real[PhiAxis]*E[ZAxis] /
        //                       x[RAxis]) * cos_v
        //                        ;
        //                       break;
        //
        //                   case ZAxis:// z
        //                       expect = (K_real[PhiAxis]*E[RAxis]/ x[RAxis]  -
        //                       K_real[RAxis]*E[PhiAxis] ) * cos_v ;
        //                       expect -= sin_v*E[PhiAxis]  / x[RAxis];
        //                       break;
        //
        //               }
        //#	else
        expect =
            (K_real[(n + 1) % 3] * E[(n + 2) % 3] - K_real[(n + 2) % 3] * E[(n + 1) % 3]) * cos_v;
        //#endif
        f1b[s] = expect;
        variance += abs((f1[s] - expect) * (f1[s] - expect));
        average += (f1[s] - expect);

        EXPECT_LE(abs(f1[s] - expect), error) << expect << "," << f1[s] << "[" << (s.x >> 1) << ","
                                              << (s.y >> 1) << "," << (s.z >> 1) << "]"
                                              << std::endl;
    });
    //#ifndef NDEBUG
    //
    //    //io::cd("/curl2/");
    //    LOGGER << SAVE(f2) << std::endl;
    //    LOGGER << SAVE(f1) << std::endl;
    //    LOGGER << SAVE(f1b) << std::endl;
    //#endif

    ASSERT_LE(std::sqrt(variance / m_p->range(mesh::SP_ES_NOT_SHARED, EDGE).size()), error);
    ASSERT_LE(abs(average / m_p->range(mesh::SP_ES_NOT_SHARED, EDGE).size()), error);
}

TEST_P(FETLTest, identity_curl_grad_f0_eq_0) {
    field_type<VERTEX> f0(m);
    field_type<EDGE> f1(m);
    field_type<FACE> f2a(m);
    field_type<FACE> f2b(m);

    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    Real mean = 0.0;
    f0.clear();
    f0.assign([&](entity_id const& s) {
        auto a = uniform_dist(gen);
        mean += a * a;
        return one * a;
    });

    mean = std::sqrt(mean) * abs(one);

    LOG_CMD(f1 = grad(f0));
    LOG_CMD(f2a = curl(f1));
    LOG_CMD(f2b = curl(grad(f0)));

    Real variance_a = 0;
    Real variance_b = 0;

    m_p->range(mesh::SP_ES_OWNED, FACE).foreach ([&](entity_id const& s) {
        variance_a += abs(f2a[s]);
        variance_b += abs(f2b[s]);
    });
    variance_a /= mean;
    variance_b /= mean;
    ASSERT_LE(std::sqrt(variance_b), error);
    ASSERT_LE(std::sqrt(variance_a), error);
}

TEST_P(FETLTest, identity_curl_grad_f3_eq_0) {
    field_type<VOLUME> f3(m);
    field_type<EDGE> f1a(m);
    field_type<EDGE> f1b(m);
    field_type<FACE> f2(m);
    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    Real mean = 0.0;

    f3.clear();
    f3.assign([&](entity_id const& s) {
        auto a = uniform_dist(gen);
        mean += a * a;
        return a * one;
    });

    mean = std::sqrt(mean) * abs(one);

    LOG_CMD(f2 = grad(f3));
    LOG_CMD(f1a = curl(f2));
    LOG_CMD(f1b = curl(grad(f3)));

    size_t count = 0;
    Real variance_a = 0;
    Real variance_b = 0;
    m_p->range(mesh::SP_ES_OWNED, EDGE).foreach ([&](entity_id const& s) {
        variance_a += abs(f1a[s]);
        variance_b += abs(f1b[s]);
    });

    variance_a /= mean;
    variance_b /= mean;
    ASSERT_LE(std::sqrt(variance_b), error);
    ASSERT_LE(std::sqrt(variance_a), error);
}

TEST_P(FETLTest, identity_div_curl_f1_eq0) {
    field_type<EDGE> f1(m);
    field_type<FACE> f2(m);
    field_type<VERTEX> f0a(m);
    field_type<VERTEX> f0b(m);

    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    f2.clear();
    Real mean = 0.0;
    f2.assign([&](entity_id const& s) {
        auto a = uniform_dist(gen);
        mean += a * a;
        return one * a;
    });

    mean = std::sqrt(mean) * abs(one);

    LOG_CMD(f1 = curl(f2));
    LOG_CMD(f0a = diverge(f1));
    LOG_CMD(f0b = diverge(curl(f2)));

    Real variance_a = 0;
    Real variance_b = 0;

    m_p->range(mesh::SP_ES_OWNED, VERTEX).foreach ([&](entity_id const& s) {
        variance_b += abs(f0b[s] * f0b[s]);
        variance_a += abs(f0a[s] * f0a[s]);
    });

    ASSERT_LE(std::sqrt(variance_b / m_p->range(mesh::SP_ES_OWNED, VERTEX).size()), error);
    ASSERT_LE(std::sqrt(variance_a / m_p->range(mesh::SP_ES_OWNED, VERTEX).size()), error);
}

TEST_P(FETLTest, identity_div_curl_f2_eq0) {
    field_type<EDGE> f1(m);
    field_type<FACE> f2(m);
    field_type<VOLUME> f3a(m);
    field_type<VOLUME> f3b(m);

    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    f1.clear();

    Real mean = 0.0;
    size_type count = 0;
    f1.assign([&](entity_id const& s) {
        auto a = uniform_dist(gen);
        mean += a * a;
        return one * a;
    });

    mean = std::sqrt(mean) * abs(one);

    LOG_CMD(f2 = curl(f1));
    LOG_CMD(f3a = diverge(f2));
    LOG_CMD(f3b = diverge(curl(f1)));

    Real variance_a = 0;
    Real variance_b = 0;
    m_p->range(mesh::SP_ES_OWNED, VOLUME).foreach ([&](entity_id const& s) {
        variance_a += abs(f3a[s] * f3a[s]);
        variance_b += abs(f3b[s] * f3b[s]);
    });

    EXPECT_LE(std::sqrt(variance_b / m_p->range(mesh::SP_ES_OWNED, VOLUME).size()), error);
    ASSERT_LE(std::sqrt(variance_a / m_p->range(mesh::SP_ES_OWNED, VOLUME).size()), error);
}

#endif /* CORE_FIELD_VECTOR_CALCULUS_TEST_H_ */
