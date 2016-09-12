//
// Created by salmon on 16-7-26.
//
#include "sp_lite_def.h"

#include "spObject.h"


static int global_obj_id_count = 0;

int spObjectCreate(spObject **obj, size_t s_in_byte)
{
    *obj = (spObject *) malloc(s_in_byte);

    (*obj)->id = global_obj_id_count;

    ++global_obj_id_count;

    return SP_SUCCESS;
};

int spObjectDestroy(spObject **obj)
{
    if (obj != NULL && *obj != NULL)
    {
        free((void *) *obj);
        *obj = NULL;
    }
    return SP_SUCCESS;
};

int spObjectId(spObject const *f) { return f->id; };