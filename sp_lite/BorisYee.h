//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_BORISYEE_H
#define SIMPLA_BORISYEE_H

#include "spParticle.h"
#include "spMesh.h"
#include "spField.h"


typedef struct boris_data_s
{
    SP_PARTICLE_DATA_HEAD

    Real *vx;
    Real *vy;
    Real *vz;

    Real *f;
    Real *w;
} boris_data;

void spBorisYeeInitializeParticle(spParticle *pg, size_type NUM_OF_PIC);

void spBorisYeeUpdateParticle(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho,
                              spField *fJ);

void spUpdateField_Yee(spMesh *ctx, Real dt, const spField *fRho, const spField *fJ, spField *fE, spField *fB);

#endif //SIMPLA_BORISYEE_H
