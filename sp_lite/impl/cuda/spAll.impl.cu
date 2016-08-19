//
// Created by salmon on 16-8-14.
//

extern "C"
{
#include "spParallelCUDA.h"
#include "../spMisc.impl.h"
#include "../spFDTD.impl.h"
#include "../spPICBoris.impl.h"

__constant__ spContext sp_ctx_d;

spContext sp_ctx;

}