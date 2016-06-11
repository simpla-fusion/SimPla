//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_BORIS_H
#define SIMPLA_BORIS_H

#include "../../src/particle/ParticleEngine.h"
#include "../../src/particle/SmallObjPool.h"


#ifdef __cplusplus
extern "C" {
#endif
struct boris_point_s
{
    POINT_HEAD;

    double v;
    double f;
    double w;
};
struct coRectMesh_s
{
    double x_lower[3], x_upper[3];
    long i_lower[3], i_upper[3];
};

void spBorisProject(struct boris_point_s const *p, double *res, struct coRectMesh_s const *m);

void spBorisLift(double *res, struct boris_point_s const *p, struct coRectMesh_s const *m);

void spBorisPushN(struct spPage *p, double dt, double const *E, double const *B, struct coRectMesh_s const *m);

void spBorisGatherN(double *res, struct spPage const *p, struct coRectMesh_s const *m);

void spBorisGatherVN(double **res, struct spPage const *p, struct coRectMesh_s const *m);

#ifdef __cplusplus
};

#include "../../src/particle/Particle.h"
#include "../../src/gtl/type_traits.h"
#include "../../src/data_model/DataType.h"

namespace simpla{namespace traits{
template<>struct type_id<boris_point_s,void>{

SP_DEFINE_STRUCT_DESCRIBE(boris_point_s ,
    double, v,
    double, f,
    double, w
           );
     };}}
namespace simpla
{


struct Boris
{
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
    void push(spPage *p0, Real dt, TE const &E1, TB const &B0) const
    {
//        spBorisPushN(p0, dt);
    }

  template<typename ...Others>
    void gather(Real *res, point_s const &p0,point_type const &x0, Others &&...) const
    {
//        spBorisGatherN(res,p0 );

    }

    template<typename ...Others>
    void gather(Vec3 *res, point_s  const &p0,  point_type const &x0, Others &&...) const
    {
//      spBorisGatherVN(&(res)[0],p0 );

    }


    template<typename ...Others>
    void gather(Real *res, spPage *p0,point_type const &x0, Others &&...) const
    {
//        spBorisGatherN(res,p0 );

    }

    template<typename ...Others>
    void gather(Vec3 *res, spPage *p0,  point_type const &x0, Others &&...) const
    {
//      spBorisGatherVN(&(res)[0],p0 );

    }
};
}//namespace simpla
#endif //__cplusplus
#endif //SIMPLA_BORIS_H
