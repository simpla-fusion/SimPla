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
#include "../ParticleProxy.h"
#include "../ParticleEngine.h"
#include "../ParticleGenerator.h"
#include "../ParticleConstraint.h"
#include "../ParticleTracker.h"

namespace simpla { namespace particle { namespace engine
{
template<typename TM>
struct BorisEngine
{
    typedef typename TM::point_type point_type;
    typedef typename TM::vector_type vector_type;


    virtual Properties &properties() = 0;

    virtual Properties const &properties() const = 0;

    DEFINE_PROPERTIES(Real, mass);

    DEFINE_PROPERTIES(Real, charge);

    DEFINE_PROPERTIES(Real, temperature);

    SP_DEFINE_STRUCT(sample_type, Vec3, x, Vec3, v, Real, f, Real, w);

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
        return sample_type{x,
                v,
                fun(x, v),
                0};
    }


    void integral(point_type const &x, sample_type const &p, Real *f) const { *f = p.f * p.w; }

    void integral(point_type const &x, sample_type const &p, nTuple<Real, 3> *v) const { *v = p.v * p.f * p.w; }

    template<typename TE, typename TB>
    void push(sample_type *p, Real dt, Real t, TE const &E, TB const &B) const
    {
        p->x += p->v * dt * 0.5;
        p->v += E(p->x) * dt;
        p->x += p->v * dt * 0.5;
    };

};

}}}//namespace simpla { namespace particle { namespace engine
namespace simpla { namespace particle
{
template<typename TM> using BorisParticle =
Particle<particle::engine::BorisEngine<TM>, TM,
        manifold::policy::FiniteVolume,
        manifold::policy::LinearInterpolator
>;

template<typename TM> using BorisTrackingParticle =
Particle<enable_tracking<particle::engine::BorisEngine<TM>>, TM,
        manifold::policy::FiniteVolume,
        manifold::policy::LinearInterpolator
>;
}}

#endif //SIMPLA_PIC_BORIS_H
