//
// Created by salmon on 16-6-12.
//

#ifndef SIMPLA_BORISINTERFACE_H_H
#define SIMPLA_BORISINTERFACE_H_H

#include "BorisYee.h"
#include "../../src/sp_config.h"
#include "../../src/gtl/Properties.h"
#include "../../src/data_model/DataType.h"
#include "../../src/particle/Particle.h"


namespace simpla { namespace traits
{
template<> struct type_id<boris_point_s, void>
{

    SP_DEFINE_PARTICLE_DESCRIBE(boris_point_s,
                                double[3], v,
                                double, f,
                                double, w
    );
};
}}//namespace simpla { namespace traits

namespace simpla { namespace particle { namespace engine
{
using namespace mesh;

template<typename TM>
struct BorisYeeCXXWrap
{

    typedef TM mesh_type;

    typedef boris_point_s point_s;


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
        spPush<point_s>(m_charge_ / m_mass_, pg, dt, E1.get(), B0.get(), &dx[0]);
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
//        spGather<point_s>(DENSITY, res, pg, std::get<0>(z), &std::get<1>(z)[0]);
    }

};
}}}//namespace simpla { namespace particle { namespace engine

namespace simpla { namespace particle
{

template<typename TM> using BorisParticle= DefaultParticle<particle::engine::BorisYeeCXXWrap<TM>, TM>;

}}//namespace simpla{namespace particle{
#endif //SIMPLA_BORISINTERFACE_H_H
