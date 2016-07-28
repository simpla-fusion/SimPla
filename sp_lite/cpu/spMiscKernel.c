//
// Created by salmon on 16-7-28.
//

#include "../spParallel.h"
#include "../spMisc.h"
#include <math.h>

void spFieldAssignValueSinKernel(size_type const *block,
                                 size_type const *thread,
                                 Real *data,
                                 size_type const *strides,
                                 Real const *k_dx,
                                 Real const *alpha0,
                                 Real amp)
{

    for (int x = 0; x < block[0]; ++x)
        for (int y = 0; y < block[1]; ++y)
            for (int z = 0; z < block[2]; ++z)
            {
                size_type s = x * strides[0] + y * strides[1] + z * strides[2];

                data[s] = amp *
                    cos(k_dx[0] * Real(x) + alpha0[0]) *
                    cos(k_dx[1] * Real(y) + alpha0[1]) *
                    cos(k_dx[2] * Real(z) + alpha0[2]);
            }
};
