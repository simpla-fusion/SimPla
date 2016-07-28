//
// Created by salmon on 16-7-26.
//
#include "sp_lite_def.h"

#include "spObject.h"

#include "spParallel.h"

#include "../src/sp_capi.h"

int spObjectCreate(spObject **obj, size_t s_in_byte)
{
    spParallelHostAlloc((void **) obj, s_in_byte);

    *obj = (spObject *) malloc(s_in_byte);

    (*obj)->id = spMPIGenerateObjectId();

    return SP_SUCCESS;
};
int spObjectDestroy(spObject **obj)
{
    if (obj != NULL && *obj != NULL) { spParallelHostFree((void *) *obj); }
    return SP_SUCCESS;
};
size_type spObjectId(spObject const *f) { return f->id; };