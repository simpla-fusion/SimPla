//
// Created by salmon on 16-7-27.
//

#ifndef SIMPLA_SPMISC_H
#define SIMPLA_SPMISC_H

#include "sp_lite_def.h"
#include "spField.h"

int spFieldAssignValueSin(spField *, Real const *k, Real const *amp);
void spFieldAssignValueSinKernel(size_type const *block,
                                 size_type const *thread,
                                 Real *data,
                                 size_type const *strides,
                                 Real const *k_dx,
                                 Real const *alpha0,
                                 Real amp);

#endif //SIMPLA_SPMISC_H
