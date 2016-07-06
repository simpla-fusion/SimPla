/*
 * spObject.c
 *
 *  Created on: 2016年7月6日
 *      Author: salmon
 */
#include "sp_lite_def.h"
#include "spObject.h"

void spFree(void **p)
{
	if (p != 0x0 && *p != 0x0)
	{
#ifdef __CUDACC__
		if (sp_is_device_ptr(*p))
		{
			CUDA_CHECK_RETURN(cudaFree(*p));
		}
		else
#endif//#ifdef __CUDA_C__
		{
			free(*p);
		}
		*p = 0x0;
	}

}

void spObjectCreate(struct spObject_s ** obj, size_t size_in_byte)
{
	*obj = (spObject *) malloc(size_in_byte);
	(*obj)->size_in_byte = size_in_byte;

}
void spObjectDestroy(spObject ** obj)
{
	if (obj != 0x0 && *obj != 0x0)
	{
		free(*obj);
	}
	*obj = 0x0;
}
