//
// Created by salmon on 16-7-26.
//
#include "sp_lite_config.h"
#include "spObject.h"
#include "spParallel.h"
#include "spMPI.h"

int spObjectCreate(spObject **obj, size_t s_in_byte)
{
    SP_CALL(spMemoryHostAlloc((void **) obj, s_in_byte));
    (*obj)->id = spMPIGenerateObjectId();
    return SP_SUCCESS;
};

int spObjectDestroy(spObject **obj)
{

    if (obj != NULL && *obj != NULL) {SP_CALL(spMemoryHostFree((void *) obj)); }
    return SP_SUCCESS;
};

size_type spObjectId(spObject const *f) { return f->id; };