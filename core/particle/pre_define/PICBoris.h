/**
 * @file PICBoris.h
 * @author salmon
 * @date 2015-11-29.
 */

#ifndef SIMPLA_PIC_BORIS_H
#define SIMPLA_PIC_BORIS_H

#include "../../manifold/policy/FvmStructuredPolicy.h"
#include "../../manifold/policy/LinearInterpolatorPolicy.h"

#include "../Particle.h"
#include "../ParticleContainer.h"
#include "../ParticleEngine.h"

namespace simpla { namespace particle { namespace engine
{

inline void BorisPusher(Real cmr, Real dt, Vec3 const &B, Vec3 const &E,
                        ::simpla::geometry::CartesianMetric::point_type *x,
                        ::simpla::geometry::CartesianMetric::vector_type *v)
{
    (*x) += (*v) * dt * 0.5;

    Vec3 v_, t;

    t = B * (cmr * dt * 0.5);

    (*v) += E * (cmr * dt * 0.5);

    v_ = (*v) + cross((*v), t);

    (*v) += cross(v_, t * 2.0) / (inner_product(t, t) + 1.0);


    (*v) += E * (cmr * dt * 0.5);

    (*x) += (*v) * dt * 0.5;
}

//inline void BorisPusher(::simpla::geometry::CylindricalMetric const &, Real cmr, Real dt, Vec3 const &B, Vec3 const &E,
//                        ::simpla::geometry::CylindricalMetric::point_type *x,
//                        ::simpla::geometry::CylindricalMetric::vector_type *v)
//{
//    (*x) += (*v) * dt * 0.5;
//
//    Vec3 v_, t;
//
//    t = B * (cmr * dt * 0.5);
//
//    (*v) += E * (cmr * dt * 0.5);
//
//    v_ = (*v) + cross((*v), t);
//
//    (*v) += cross(v_, t * 2.0) / (inner_product(t, t) + 1.0);
//
//
//    (*v) += E * (cmr * dt * 0.5);
//
//    (*x) += (*v) * dt * 0.5;
//}

template<typename TM> struct BorisEngine;

template<typename TM>
struct BorisEngine
{

    typedef TM mesh_type;
    typedef typename mesh_type::scalar_type scalar_type;
    typedef typename mesh_type::point_type point_type;
    typedef typename mesh_type::vector_type vector_type;

    typedef traits::field_t<scalar_type, mesh_type, EDGE> E_field;
    typedef traits::field_t<scalar_type, mesh_type, FACE> B_field;

    E_field E1;
    B_field B0;

    mesh_type &m_mesh_;

    BorisEngine(mesh_type &m) : m_mesh_(m) { }

    ~BorisEngine() { }

    virtual Properties &properties() = 0;

    virtual Properties const &properties() const = 0;

    DEFINE_PROPERTIES(Real, mass);

    DEFINE_PROPERTIES(Real, charge);

    DEFINE_PROPERTIES(Real, temperature);

    SP_DEFINE_STRUCT(sample_type, size_t, _tag, point_type, x, vector_type, v, Real, f, Real, w);

    void deploy()
    {
        m_mass_ = properties()["mass"].template as<Real>(1.0);
        m_charge_ = properties()["charge"].template as<Real>(1.0);
        m_temperature_ = properties()["temperature"].template as<Real>(1.0);

        if (E1.empty())
        {
            E_field{m_mesh_, properties()["E"].template as<std::string>("E")}.swap(E1);
        }
        if (B0.empty())
        {
            B_field(m_mesh_, properties()["E"].template as<std::string>("B")).swap(B0);
        }
    }


    point_type project(sample_type const &z) const { return z.x; }

    std::tuple<point_type, vector_type> push_forward(sample_type const &z) const
    {
        return std::forward_as_tuple(z.x, z.v);
    }


    sample_type lift(point_type const &x, vector_type const &v, Real f = 0) const
    {
        sample_type res;

        res.x = x;

        res.v = v;

        res.f = f;

        res.w = 0;

        return std::move(res);
    }

    sample_type sample(point_type const &x, vector_type const &v, Real f) const { return sample_type{x, v, f, 1.0}; }

    template<typename TFunc>
    sample_type lift(point_type const &x, vector_type const &v, TFunc const &fun) const
    {
        return sample_type{0U, x, v, fun(x, v), 0};
    }

    void push(Real t0, Real t1, sample_type *p0) const
    {
        Real dt = t1 - t0;

        auto E = E1(project(*p0));

        auto B = B0(project(*p0));


        Real cmr = m_charge_ / m_mass_;

        p0->x += p0->v * dt * 0.5;

        Vec3 v_, t;

        t = B * (cmr * dt * 0.5);

        p0->v += E * (cmr * dt * 0.5);

        v_ = p0->v + cross(p0->v, t);

        p0->v += cross(v_, t * 2.0) / (inner_product(t, t) + 1.0);

        p0->v += E * (cmr * dt * 0.5);

        p0->x += p0->v * dt * 0.5;
    }

    void integral(point_type const &x0, sample_type const &p, scalar_type *res) const
    {

    }

    void integral(point_type const &x0, sample_type const &p, vector_type *res) const
    {

    }
//
//
//    void integral(point_type const &x, sample_type const &p, Real *f) const
//    {
//        *f = p.f * p.w;
//    }
//
//    void integral(point_type const &x, sample_type const &p, nTuple<Real, 3> *v) const
//    {
//        *v = p.v * p.f * p.w;
//    }
//
//
//    template<typename TE, typename TB>
//    void push(sample_type *p0, Real dt, Real t0, TE const &fE, TB const &fB) const
//    {
//
//
//        vector_type E, B;
//
//        E = fE(project(*p0));
//
//        B = fB(project(*p0));
//
//
//        Real cmr = m_charge_ / m_mass_;
//
//        p0->x += p0->v * dt * 0.5;
//
//        Vec3 v_, t;
//
//        t = B * (cmr * dt * 0.5);
//
//        p0->v += E * (cmr * dt * 0.5);
//
//        v_ = p0->v + cross(p0->v, t);
//
//        p0->v += cross(v_, t * 2.0) / (inner_product(t, t) + 1.0);
//
//
//        p0->v += E * (cmr * dt * 0.5);
//
//        p0->x += p0->v * dt * 0.5;
//
//
//    };
};
//
//
//template<>
//struct BorisEngine<geometry::CylindricalMetric>
//{
//    typedef typename geometry::CylindricalMetric::point_type point_type;
//    typedef typename geometry::CylindricalMetric::vector_type vector_type;
//
//
//    virtual Properties &properties() = 0;
//
//    virtual Properties const &properties() const = 0;
//
//    DEFINE_PROPERTIES(Real, mass);
//
//    DEFINE_PROPERTIES(Real, charge);
//
//    DEFINE_PROPERTIES(Real, temperature);
//
//    SP_DEFINE_STRUCT(sample_type, point_type, x, vector_type, v, Real, f, Real, w);
//
//    void deploy()
//    {
//        m_mass_ = properties()["mass"].template as<Real>(1.0);
//        m_charge_ = properties()["charge"].template as<Real>(1.0);
//        m_temperature_ = properties()["temperature"].template as<Real>(1.0);
//    }
//
//    point_type project(sample_type const &z) const { return z.x; }
//
//    std::tuple<point_type, vector_type> push_forward(sample_type const &z) const
//    {
//        return std::forward_as_tuple(z.x, z.v);
//    }
//
//
//    sample_type lift(point_type const &x, vector_type const &v, Real f = 0) const { return sample_type{x, v, f, 1.0}; }
//
//    sample_type sample(point_type const &x, vector_type const &v, Real f) const { return sample_type{x, v, f, 1.0}; }
//
//    template<typename TFunc>
//    sample_type lift(point_type const &x, vector_type const &v, TFunc const &fun) const
//    {
//        return sample_type{x, v, fun(x, v), 0};
//    }
//
//
//    void integral(point_type const &x, sample_type const &p, Real *f) const { *f = p.f * p.w; }
//
//    void integral(point_type const &x, sample_type const &p, nTuple<Real, 3> *v) const { *v = p.v * p.f * p.w; }
//
//
//    template<typename TE, typename TB>
//    void push(sample_type *p0, Real dt, Real t0, TE const &fE, TB const &fB) const
//    {
//
//        vector_type E, B;
//
//        E = fE(project(*p0));
//
//        B = fB(project(*p0));
//
//        Real cmr = m_charge_ / m_mass_;
//
//        Real r = p0->x[0];
//
//
//        p0->x[0] += p0->v[0] * dt * 0.5;
//        p0->x[1] += p0->v[1] * dt * 0.5;
//        p0->x[2] += p0->v[2] * dt * 0.5;
//
//        Vec3 v_, t;
//
//        t = B * (cmr * dt * 0.5);
//
//        p0->v += E * (cmr * dt * 0.5);
//
//        v_ = p0->v + cross(p0->v, t);
//
//        p0->v += cross(v_, t * 2.0) / (inner_product(t, t) + 1.0);
//
//
//        p0->v += E * (cmr * dt * 0.5);
//
//
//        p0->x[0] += p0->v[0] * dt * 0.5;
//        p0->x[1] += p0->v[1] * dt * 0.5;
//        p0->x[2] += p0->v[2] * dt * 0.5;
//
//
//    };
//};
}}}//namespace simpla { namespace particle { namespace engine
namespace simpla { namespace particle
{
template<typename TM> using BorisParticle = ::simpla::particle::Particle<engine::BorisEngine<TM>>;


}}

#endif //SIMPLA_PIC_BORIS_H
