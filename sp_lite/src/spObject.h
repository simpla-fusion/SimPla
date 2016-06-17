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
	struct spObject_s * self;
	size_type size_in_byte;
	byte_type __others[];
} spObject;
#define SP_OBJECT_HEAD	void * self;size_type size_in_byte;

MC_HOST extern inline void spFree(void **p)
{
#if !defined(__CUDA_ARCH__)
	if (p != 0x0 && sp_is_device_ptr(*p))
	{
		CUDA_CHECK_RETURN(cudaFree(*p));
	}
	else
#endif
	{
		free(*p);
	}
	*p = 0x0;
}

MC_HOST extern inline void spCreateObject(spObject ** obj, size_type size_in_byte)
{
	CUDA_CHECK(size_in_byte);

	*obj = (spObject *) malloc(size_in_byte);
	(*obj)->size_in_byte = size_in_byte;
	(*obj)->self = 0x0;
	CUDA_CHECK((*obj)->size_in_byte);

}
MC_HOST extern inline void spDestroyObject(spObject ** obj)
{
	if (*obj == 0x0)
	{
		return;
	}

	spDestroyObject(&((*obj)->self));

	spFree((void**) obj);

}
MC_HOST extern inline void spObjectHostToDevice(spObject * obj)
{
	spObject * tmp = obj->self;
	if (tmp == 0x0)
	{
		CUDA_CHECK(obj->size_in_byte);

		CUDA_CHECK_RETURN(cudaMalloc(&(tmp), obj->size_in_byte));
	}
	CUDA_CHECK_RETURN(cudaMemcpy(tmp, obj, obj->size_in_byte, cudaMemcpyDefault));
	obj->self = tmp;
}
MC_HOST inline void spObjectDeviceToHost(spObject * obj)
{
	if (obj != 0x0 && obj->self != 0x0)
	{
		spObject * tmp = obj->self;
		CUDA_CHECK_RETURN(cudaMemcpy(obj, obj->self, obj->size_in_byte, cudaMemcpyDefault));
		obj->self = tmp;
	}
}
#endif /* SPOBJECT_H_ */
