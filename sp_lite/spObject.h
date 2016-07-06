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

extern inline void spFree(void **p)
{
	if (p != 0x0 && *p != 0x0)
	{

		if (sp_is_device_ptr(*p))
		{
			CUDA_CHECK_RETURN(cudaFree(*p));
		}
		else
		{
			free(*p);
		}
		*p = 0x0;
	}

}

extern inline void spObjectCreate(spObject ** obj, size_type size_in_byte)
{
	*obj = (spObject *) malloc(size_in_byte);
	(*obj)->size_in_byte = size_in_byte;

}
extern inline void spObjectDestroy(spObject ** obj)
{
	if (obj != 0x0 && *obj != 0x0)
	{
		free(*obj);
	}
	*obj = 0x0;
}

#endif /* SPOBJECT_H_ */
