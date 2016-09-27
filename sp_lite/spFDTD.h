//
// Created by salmon on 16-7-28.
//

#ifndef SIMPLA_FDTD_H
#define SIMPLA_FDTD_H

#include "sp_lite_config.h"
#include "spPhysicalConstants.h"

struct spField_s;
struct spMesh_s;

int spFDTDInitialValueSin(spField *, Real const *k, Real const *amp);

int spFDTDUpdate(Real dt,
                 const struct spField_s *fRho,
                 const struct spField_s *fJ,
                 struct spField_s *fE,
                 struct spField_s *fB);

int spFDTDDiv(const spField *fJ, spField *fRho);

int spFDTDMultiplyByScalar(spField *fRho, Real a);


#endif //SIMPLA_FDTD_H
