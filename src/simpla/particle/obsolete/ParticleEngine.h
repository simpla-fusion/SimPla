/**
 * @file ParticleEngine.h
 *
 * @date    2014-8-29  AM10:36:23
 * @author salmon
 */

#ifndef PARTICLE_ENGINE_H_
#define PARTICLE_ENGINE_H_

#include "ParticleInterface.h"
#include <cstddef>
#include "../toolbox/DataTypeExt.h"

namespace simpla { namespace particle
{


template<typename TM, typename TP>
struct ParticleEngine
{
    typedef TM mesh_type;

    typedef TP point_s;

    virtual Properties &properties() = 0;

    virtual Properties const &properties() const = 0;

    DEFINE_PROPERTIES(Real, mass);

    DEFINE_PROPERTIES(Real, charge);

    DEFINE_PROPERTIES(Real, temperature);


    void deploy()
    {
        m_mass_ = properties()["mass"].template as<Real>(1.0);
        m_charge_ = properties()["charge"].template as<Real>(1.0);
        m_temperature_ = properties()["temperature"].template as<Real>(1.0);

    }

    virtual mesh_type const &mesh() const = 0;

    point_type project(point_s const &z) const
    {
        point_type res;
//        spBorisProject(&z,&res[0]);
        return std::move(res);
    }

    point_s lift(Real const z[6]) const
    {
        point_s res;
//        spBorisLift(&z,&res );
        return std::move(res);
    }

    template<typename TE, typename TB>
    void push(point_s *p0, Real dt, TE const &E1, TB const &B0) const
    {
//        spBorisPushN(p0, dt);
    }

    template<typename TE, typename TB>
    void push(spPage *pg, Real dt, TE const &E1, TB const &B0) const
    {
        mesh_type const &m = mesh();
        point_type dx = m.dx();
        index_tuple i_lower, i_upper;
        std::tie(i_lower, i_upper) = m.index_box();
        spPush<point_s>(m_charge_ / m_mass_, pg, dt, E1.get(), B0.get(), &i_lower[0], &i_upper[0], &dx[0]);
    }

    template<typename ...Others>
    void gather(Real *res, point_s const &p0, point_type const &x0, Others &&...) const
    {
//        spBorisGatherN(res,p0 );

    }

    template<typename ...Others>
    void gather(Vec3 *res, point_s const &p0, point_type const &x0, Others &&...) const
    {

    }


    template<typename ...Others>
    void gather(ParticleMomentType type, Real *res, spPage *pg, point_type const &x, Others &&...) const
    {
        auto z = mesh().point_global_to_local(x);
        spGather<point_s>(DENSITY, res, pg, std::get<0>(z), &std::get<1>(z)[0]);
    }


};
}} //namespace simpla{namespace particle{


#endif /* PARTICLE_ENGINE_H_ */
