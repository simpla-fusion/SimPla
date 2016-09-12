/*
 * spObject.h
 *
 *  Created on: 2016年6月17日
 *      Author: salmon
 */

#ifndef SPOBJECT_H_
#define SPOBJECT_H_
#include "sp_lite_def.h"

#define SP_OBJECT_HEAD   int id;

typedef struct spObject_s
{
    SP_OBJECT_HEAD
    byte_type __others[];
} spObject;

int spObjectCreate(spObject **obj, size_t s_in_byte);

int spObjectDestroy(spObject **obj);

int spObjectId(spObject const *f);


#endif /* SPOBJECT_H_ */
