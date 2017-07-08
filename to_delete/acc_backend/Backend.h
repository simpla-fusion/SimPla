//
// Created by salmon on 17-6-23.
//

#ifndef SIMPLA_ACCBACKEND_H
#define SIMPLA_ACCBACKEND_H
#ifdef __CUDA__
#include "CUDABackend.h"
#else
#include "CPUBackend.h"
#endif
#endif  // SIMPLA_ACCBACKEND_H
