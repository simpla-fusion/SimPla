//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_BORISYEE_H
#define SIMPLA_BORISYEE_H

#include "spParticle.h"
#include "spMesh.h"
#include "spField.h"

void spBorisYeeInitializeParticle(spParticle *pg, int NUM_OF_PIC);

void spBorisYeeUpdateParticle(spParticle *pg, Real dt, const spField *fE, const spField *fB, spField *fRho,
                              spField *fJ);

void spUpdateField_Yee(spMesh *ctx, Real dt, const spField *fRho, const spField *fJ, spField *fE, spField *fB);

#endif //SIMPLA_BORISYEE_H
