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

struct BorisEngine : public base::Object
{

    HAS_PROPERTIES

    virtual std::ostream &print(std::ostream &os, int indent) const
    {
        m_properties_.print(os, indent);
        return os;
    }

    DEFINE_PROPERTIES(Real, mass);

    DEFINE_PROPERTIES(Real, charge);

    DEFINE_PROPERTIES(Real, temperature);


    SP_DEFINE_STRUCT(point_type, Vec3, x, Vec3, v, Real, f, Real, w);

    template<typename TDict> void load(TDict const &dict)
    {
        mass(dict["mass"].template as<Real>(1.0));
        charge(dict["charge"].template as<Real>(1.0));
        temperature(dict["temperature"].template as<Real>(1.0));

        touch();
    }

    Vec3 project(point_type const &z) const { return z.x; }

    std::tuple<Vec3, Vec3> push_forward(point_type const &z) const
    {
        return std::forward_as_tuple(z.x, z.v);
    }


    point_type lift(Vec3 const &x, Vec3 const &v, Real f = 0) const
    {
        return point_type{x, v, f, 1.0};
    }

    point_type sample(Vec3 const &x, Vec3 const &v, Real f) const
    {
        return point_type{x, v, f, 1.0};
    }

    template<typename TFunc>
    point_type lift(Vec3 const &x, Vec3 const &v, TFunc const &fun) const
    {
        return point_type{x, v, fun(x, v), 0};
    }


    void integral(Vec3 const &x, point_type const &p, Real *f) const
    {
        *f = p.f * p.w;
    }

    void integral(Vec3 const &x, point_type const &p, nTuple<Real, 3> *v) const
    {
        *v = p.v * p.f * p.w;
    }

    template<typename TE, typename TB>
    void push(point_type *p, Real dt, Real t, TE const &E, TB const &B) const
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
Particle<particle::engine::BorisEngine, TM,
        manifold::policy::FiniteVolume,
        manifold::policy::LinearInterpolator
>;

template<typename TM> using BorisTestParticle =
Particle<enable_tracking<particle::engine::BorisEngine>, TM,
        manifold::policy::FiniteVolume,
        manifold::policy::LinearInterpolator
>;
}}

#endif //SIMPLA_PIC_BORIS_H
