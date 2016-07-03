/*
 * spObject.h
 *
 *  Created on: 2016年6月17日
 *      Author: salmon
 */

#ifndef SPOBJECT_H_
#define SPOBJECT_H_
#include "sp_def.h"

typedef struct spObject_s
{
	struct spObject_s * device_self;
	size_type size_in_byte;
	byte_type __others[];
} spObject;
#define SP_OBJECT_HEAD	struct spObject_s * device_self;size_type size_in_byte;

MC_HOST extern inline void spFree(void **p)
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

MC_HOST extern inline void spCreateObject(spObject ** obj, size_type size_in_byte)
{
	*obj = (spObject *) malloc(size_in_byte);
	(*obj)->size_in_byte = size_in_byte;
	CUDA_CHECK_RETURN(cudaMalloc(&((*obj)->device_self), size_in_byte));

}
MC_HOST extern inline void spDestroyObject(spObject ** obj)
{
	if (obj != 0x0 && *obj != 0x0)
	{
		CUDA_CHECK_RETURN(cudaFree((*obj)->device_self));
		free(*obj);
	}
	*obj = 0x0;
}

MC_HOST extern inline void spObjectHostToDevice(spObject * obj)
{
	spObject * tmp = obj->device_self;
	obj->device_self = 0x0;
	CUDA_CHECK_RETURN(cudaMemcpy(tmp, obj, obj->size_in_byte, cudaMemcpyHostToDevice));
	obj->device_self = tmp;

}
MC_HOST inline void spObjectDeviceToHost(spObject * obj)
{
	if (obj != 0x0 && obj->device_self != 0x0)
	{
		spObject * tmp = obj->device_self;
		CUDA_CHECK_RETURN(cudaMemcpy(obj, obj->device_self, obj->size_in_byte, cudaMemcpyDefault));
		obj->device_self = tmp;
	}
}
MC_HOST extern inline spObject * spObject_device_(spObject * obj)
{
	spObjectHostToDevice(obj);

	return obj->device_self;
}
#endif /* SPOBJECT_H_ */