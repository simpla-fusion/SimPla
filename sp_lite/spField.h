/*
 * spField.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPFIELD_H_
#define SPFIELD_H_
#include "sp_def.h"
#include "spIOStream.h"
#include "spMPI.h"

#include "spObject.h"
#include "spMesh.h"

typedef struct spField_s
{
	SP_OBJECT_HEAD
	int iform;
	Real * device_data;
	Real * host_data;
} spField;

void spFieldCreate(const spMesh *ctx, spField **f, int iform);

void spFieldDestroy(spField **f);

void spFieldClear(spMesh const *mesh, spField *f);

int spFieldWrite(spMesh const *ctx, spField *f, spIOStream * file, char const name[], int flag);

int spFieldRead(spMesh const *ctx, spField **f, spIOStream * file, char const name[], int flag);

int spSyncField(spMesh const *ctx, spField *f);

#endif /* SPFIELD_H_ */
