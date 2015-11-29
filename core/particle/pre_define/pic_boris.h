/**
 * @file pic_boris.h
 * @author salmon
 * @date 2015-11-29.
 */

#ifndef SIMPLA_PIC_BORIS_H
#define SIMPLA_PIC_BORIS_H

#include "../particle_engine.h"

namespace simpla { namespace particle
{
namespace tags
{
struct Boris;
}
template<typename TM> using BorisParticle=Particle<ParticleEngine<tags::Boris>, TM>;

template<>
struct ParticleEngine<tags::Boris>
{
    SP_DEFINE_STRUCT(point_type,
                     Vec3, x,
                     Vec3, v
    );

    SP_DEFINE_STRUCT(sample_type,
                     point_type, z,
                     Real, f,
                     Real, w
    );

    SP_DEFINE_PROPERTIES(
            Real, mass,
            Real, charge,
            Real, temperature
    )

    void update() { }

    Vec3 project(sample_type const &p) const { return p.z.x; }

    Real function_value(sample_type const &p) const { return p.f * p.w; }


    point_type lift(Vec3 const &x, Vec3 const &v) const
    {
        return point_type{x, v};
    }

    point_type lift(std::tuple<Vec3, Vec3> const &z) const
    {
        return point_type{std::get<0>(z), std::get<1>(z)};
    }

    sample_type sample(Vec3 const &x, Vec3 const &v, Real f) const
    {
        return sample_type{lift(x, v), f, 1.0};
    }

    sample_type sample(point_type const &z, Real f) const
    {
        return sample_type{z, f, 0};
    }

    template<typename TFunc>
    sample_type lift(point_type const &z, TFunc const &fun) const
    {
        return sample_type{z, fun(z), 0};
    }


    void integral(Vec3 const &x, sample_type const &p, Real *f) const
    {
        *f = p.f * p.w;
    }

    void integral(Vec3 const &x, sample_type const &p, nTuple<Real, 3> *v) const
    {
        *v = p.z.v * p.f * p.w;
    }

    template<typename TE, typename TB>
    void push(sample_type *p, Real dt, Real t, TE const &E, TB const &B)
    {
        p->z.x += p->z.v * dt * 0.5;
        p->z.v += E(p->z.x) * dt;
        p->z.x += p->z.v * dt * 0.5;
    };

};

}}//namespace simpla { namespace particle

#endif //SIMPLA_PIC_BORIS_H
