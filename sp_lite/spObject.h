/*
 * spObject.h
 *
 *  Created on: 2016年6月17日
 *      Author: salmon
 */

#ifndef SPOBJECT_H_
#define SPOBJECT_H_
#include "sp_lite_def.h"

typedef struct spObject_s
{
	size_type size_in_byte;
	byte_type __others[];
} spObject;

#define SP_OBJECT_HEAD	 size_type size_in_byte;

void spFree(void **p);
void spObjectCreate(spObject ** obj, size_t s_in_byte);
void spObjectDestroy(spObject ** obj);

#endif /* SPOBJECT_H_ */
