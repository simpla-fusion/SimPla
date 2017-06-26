//
// Created by salmon on 17-6-23.
//

#ifndef SIMPLA_HOST_DEFINE_H
#define SIMPLA_HOST_DEFINE_H

#ifndef __CUDA__
#define __host__
#define __device__
#define __managed__
#else
#include </usr/local/cuda/include/driver_types.h>
#include </usr/local/cuda/include/device_launch_parameters.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>
#endif
#endif  // SIMPLA_HOST_DEFINE_H
