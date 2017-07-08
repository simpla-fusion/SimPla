/**
 * @file PICGyro.h
 * @author salmon
 * @date 2016-01-13.
 */

#ifndef SIMPLA_PIC_GYRO_H
#define SIMPLA_PIC_GYRO_H

#include "simpla/geometry/FVMStructured.h"
#include "simpla/geometry/LinearInterpolator.h"

#include "../../toolbox/DataTypeExt.h"
#include "simpla/particle/ParticleOld.h"
#include "simpla/particle/ParticleContainer.h"
#include "simpla/particle/ParticleEngine.h"
#include "simpla/geometry/csCylindrical.h"
#include "../../manifold/pre_define/PreDefine.h"

namespace simpla { namespace particle { namespace engine
{

template<typename TM> struct GyroEngine;

template<typename TM>
struct GyroParticleWithCylindricalCoord
{

    static_assert(std::is_same<typename TM::metric_type, ::simpla::geometry::CylindricalMetric>::value,
                  "Mesh is not Cylindrical");


    typedef TM mesh_type;
    typedef typename mesh_type::scalar_type scalar_type;
    typedef typename mesh_type::point_type point_type;
    typedef typename mesh_type::vector_type vector_type;


    virtual Properties &properties() = 0;

    virtual Properties const &properties() const = 0;

    DEFINE_PROPERTIES(Real, mass);

    DEFINE_PROPERTIES(Real, charge);

    DEFINE_PROPERTIES(Real, temperature);

    SP_DEFINE_STRUCT(sample_type, size_t, _tag,
                     point_type, X, vector_type, V,
                     Real, f, Real, w);
private:
    mesh_type &m_mesh_;

public:
    typedef traits::field_t<scalar_type, mesh_type, VERTEX> scalar_field;

    typedef traits::field_t<scalar_type, mesh_type, EDGE> E_field;

    typedef traits::field_t<scalar_type, mesh_type, FACE> B_field;

    typedef traits::field_t<scalar_type, mesh_type, VERTEX> n_field;


    scalar_field n0;
    scalar_field BB;

    E_field E1;
    B_field B1;
    B_field B0;

private:

    Real m_inv_cmr_;
    Real m_cmr_;
public:

    GyroParticleWithCylindricalCoord(mesh_type &m) : m_mesh_(m) { }

    virtual ~GyroParticleWithCylindricalCoord() { }

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

        if (B1.empty())
        {
            B_field(m_mesh_, properties()["E"].template as<std::string>("B1")).swap(B1);
        }

        if (BB.empty())
        {
            BB = dot(B0, B0);
        }

        m_cmr_ = m_charge_ / m_mass_;
        m_inv_cmr_ = 1.0 / m_cmr_;
    }

    point_type project(sample_type const &z) const { return z.X; }

    sample_type lift(point_type const &x, vector_type const &v, Real f = 0) const
    {
        sample_type res;
//        scalar_type BB_ = BB(x);
//        vector_type Bv_ = B0(x);
//
//
//        vector_type r = -cross(v, Bv_) / BB_ * m_inv_cmr_;
//
//        res.X = x - r;
//
//        res.u = dot(v, Bv_) / BB_;
//
//        res.mu = m_mass_ * (dot(v, v) - res.u * res.u) / std::sqrt(BB_);

        return std::move(res);

    }


    void integral(point_type const &x0, sample_type const &p, Real *f) const
    {
//        *f = p.f * p.w;
    }

    void integral(point_type const &x0, sample_type const &p, nTuple<Real, 3> *v) const
    {
//        *v = p.v * p.f * p.w;
    }


    void push(Real dt, Real t0, sample_type *p0) const
    {


//        vector_type E, B;
//
//        E = E1(p0->X);
//
//        B = B1(p0->X);
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


    };
};

}}}//namespace simpla { namespace particle { namespace engine
namespace simpla { namespace particle
{
template<typename TM> using GyroParticle = ParticleOld<particle::engine::GyroParticleWithCylindricalCoord<TM>, TM>;

}}
#endif //SIMPLA_PIC_GYRO_H
