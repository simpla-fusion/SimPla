/*
 * spField.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPFIELD_H_
#define SPFIELD_H_
#include "sp_def.h"
#include "spObject.h"
#include "spMesh.h"

typedef struct spField_s
{
	SP_OBJECT_HEAD
	int iform;
	Real * device_data;
	Real * host_data;
} sp_field_type;

MC_HOST void spCreateField(const spMesh *ctx, sp_field_type **f, int iform);

MC_HOST void spDestroyField(sp_field_type **f);

MC_HOST void spClearField(spMesh const *mesh, sp_field_type *f);

MC_HOST int spWriteField(spMesh const *ctx, sp_field_type *f, char const name[], int flag);

MC_HOST int spReadField(spMesh const *ctx, sp_field_type **f, char const name[], int flag);

MC_HOST int spSyncField(spMesh const *ctx, sp_field_type *f);

#endif /* SPFIELD_H_ */
