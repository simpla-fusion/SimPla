//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_BORISYEE_H
#define SIMPLA_BORISYEE_H

#include "../../sp_lite/spParticle.h"
#include "../../sp_lite/spMesh.h"
#include "../../sp_lite/spField.h"
#include "../../../sp_lite/spField.h"

void spInitializeParticle_BorisYee(spMesh *ctx, sp_particle_type *pg,
		size_type NUM_OF_PIC);

void spUpdateParticle_BorisYee(spMesh *ctx, Real dt, sp_particle_type *pg,
		const sp_field_type * fE, const sp_field_type * fB,
		sp_field_type * fRho, sp_field_type * fJ);

int spUpdateField_Yee(spMesh *ctx, Real dt, const spField_s *fRho,
                      const spField_s *fJ, spField_s *fE, spField_s *fB);

#endif //SIMPLA_BORISYEE_H
