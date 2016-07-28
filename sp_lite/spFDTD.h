//
// Created by salmon on 16-7-28.
//

#ifndef SIMPLA_FDTD_H
#define SIMPLA_FDTD_H

#include "sp_lite_def.h"

struct spMesh_s;
struct spField_s;

int spUpdateFieldFDTD(struct spMesh_s const *ctx,
                      Real dt,
                      const struct spField_s *fRho,
                      const struct spField_s *fJ,
                      struct spField_s *fE,
                      struct spField_s *fB);
#endif //SIMPLA_FDTD_H
