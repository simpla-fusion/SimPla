//
// Created by salmon on 16-7-26.
//
#include "sp_config.h"
#include "spObject.h"
#include "spParallel.h"
#include "spMPI.h"

int spObjectCreate(spObject **obj, size_t s_in_byte)
{
    int error_code = SP_SUCCESS;
    SP_CALL(spMemHostAlloc((void **) obj, s_in_byte));
    (*obj)->id = spMPIGenerateObjectId();
    return error_code;
};

int spObjectDestroy(spObject **obj)
{
    int error_code = SP_SUCCESS;
    if (obj != NULL && *obj != NULL) { SP_CALL(spParallelHostFree((void *) obj)); }
    return error_code;
};

size_type spObjectId(spObject const *f) { return f->id; };