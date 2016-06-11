//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_BORIS_H
#define SIMPLA_BORIS_H

#include "../../src/particle/ParticleInterface.h"
#include "../../src/particle/SmallObjPool.h"
#include "../../src/sp_config.h"


#ifdef __cplusplus
extern "C" {
#endif
struct boris_point_s
{
    POINT_HEAD;

    double v[3];
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

void spBorisPushN(struct spPage *p, Real cmr, double dt, Real const *E, Real const *B, const Real dx[3]);

void spBorisGatherN(double *res, struct spPage const *p, struct coRectMesh_s const *m);

void spBorisGatherVN(double **res, struct spPage const *p, struct coRectMesh_s const *m);

#ifdef __cplusplus
};



namespace simpla{namespace traits{
template<>struct type_id<boris_point_s,void>{

SP_DEFINE_PARTICLE_DESCRIBE(boris_point_s ,
    double[3], v,
    double, f,
    double, w
           );
     };
}



}

#endif //__cplusplus
#endif //SIMPLA_BORIS_H
