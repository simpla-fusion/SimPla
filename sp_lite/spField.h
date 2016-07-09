/*
 * spField.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPFIELD_H_
#define SPFIELD_H_
#include "sp_lite_def.h"
#include "spObject.h"
//#include "spMesh.h"

typedef struct spField_s
{
	SP_OBJECT_HEAD
	const struct spMesh_s * m;
	int iform;
	Real * device_data;
	Real * host_data;
} spField;

void spFieldCreate(const struct spMesh_s *ctx, spField **f, int iform);

void spFieldDestroy(spField **f);

void spFieldDeploy(spField *f);

void spFieldClear(spField *f);

//void spFieldWrite(spField *f, spIOStream * os, char const name[], int flag);
//
//void spFieldRead(spField * f, spIOStream * os, char const name[], int flag);

void spFieldSync(spField *f);

#endif /* SPFIELD_H_ */
