/**
 * @file GuidingCenter.h
 * @author salmon
 * @date 2015-12-29.
 */

#ifndef SIMPLA_GUIDINGCENTER_H
#define SIMPLA_GUIDINGCENTER_H


#include "../../manifold/policy/FvmStructuredPolicy.h"
#include "../../manifold/policy/LinearInterpolatorPolicy.h"

#include "../Particle.h"
#include "../ParticleProxy.h"
#include "../ParticleEngine.h"
#include "../ParticleGenerator.h"
#include "../ParticleConstraint.h"
#include "../ParticleTracker.h"
#include "../../geometry/csCartesian.h"
#include "../../geometry/csCylindrical.h"

namespace simpla { namespace particle { namespace engine
{

template<typename TM> struct GuidingCenterEngine;


template<>
struct GuidingCenterEngine<geometry::CylindricalMetric>
{
    typedef typename geometry::CylindricalMetric::point_type point_type;
    typedef typename geometry::CylindricalMetric::vector_type vector_type;


    virtual Properties &properties() = 0;

    virtual Properties const &properties() const = 0;

    DEFINE_PROPERTIES(Real, mass);

    DEFINE_PROPERTIES(Real, charge);

    DEFINE_PROPERTIES(Real, temperature);

    SP_DEFINE_STRUCT(sample_type, Real, R, Real, Z, Real, Phi, Real, u, Real mu, Real, f, Real, w);

    void deploy()
    {
        m_mass_ = properties()["mass"].template as<Real>(1.0);
        m_charge_ = properties()["charge"].template as<Real>(1.0);
        m_temperature_ = properties()["temperature"].template as<Real>(1.0);
    }

    point_type project(sample_type const &z) const { return z.x; }

    std::tuple<point_type, vector_type> push_forward(sample_type const &z) const
    {
        return std::forward_as_tuple(z.x, z.v);
    }


    sample_type lift(point_type const &x, vector_type const &v, Real f = 0) const { return sample_type{x, v, f, 1.0}; }

    sample_type sample(point_type const &x, vector_type const &v, Real f) const { return sample_type{x, v, f, 1.0}; }

    template<typename TFunc>
    sample_type lift(point_type const &x, vector_type const &v, TFunc const &fun) const
    {
        return sample_type{x, v, fun(x, v), 0};
    }


    void integral(point_type const &x, sample_type const &p, Real *f) const { *f = p.f * p.w; }

    void integral(point_type const &x, sample_type const &p, nTuple<Real, 3> *v) const { *v = p.v * p.f * p.w; }


    template<typename TE, typename TB>
    void push(sample_type *p0, Real dt, Real t0, TE const &fE, TB const &fB) const
    {

        vector_type E, B;

        E = fE(project(*p0));

        B = fB(project(*p0));

        Real cmr = m_charge_ / m_mass_;

        Real r = p0->x[0];


        p0->x[0] += p0->v[0] * dt * 0.5;
        p0->x[1] += p0->v[1] * dt * 0.5;
        p0->x[2] += p0->v[2] * dt * 0.5 / r;

        Vec3 v_, t;

        t = B * (cmr * dt * 0.5);

        p0->v += E * (cmr * dt * 0.5);

        v_ = p0->v + cross(p0->v, t);

        p0->v += cross(v_, t * 2.0) / (inner_product(t, t) + 1.0);


        p0->v += E * (cmr * dt * 0.5);


        p0->x[0] += p0->v[0] * dt * 0.5;
        p0->x[1] += p0->v[1] * dt * 0.5;
        p0->x[2] += p0->v[2] * dt * 0.5 / r;


    };
};
}}}//namespace simpla { namespace particle { namespace engine
namespace simpla { namespace particle
{
template<typename TM> using GuidingCenterParticle =
Particle<particle::engine::GuidingCenterEngine<typename TM::metric_type>, TM,
        manifold::policy::FiniteVolume,
        manifold::policy::LinearInterpolator
>;

template<typename TM> using GuidingCenterTrackingParticle =
Particle<enable_tracking<particle::engine::GuidingCenterEngine<typename TM::metric_type>>, TM,
        manifold::policy::FiniteVolume,
        manifold::policy::LinearInterpolator
>;
}}

#endif //SIMPLA_GUIDINGCENTER_H
