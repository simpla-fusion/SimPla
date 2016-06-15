/*
 * spField.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPFIELD_H_
#define SPFIELD_H_

#include "spMesh.h"

struct spField_s;

typedef struct spField_s sp_field_type;

void spCreateField(const spMesh *ctx, sp_field_type **f, int iform);

void spDestroyField(const spMesh *ctx, sp_field_type **f);

int spWriteField(spMesh const *ctx, sp_field_type *f, char const name[], int flag);

int spReadField(spMesh const *ctx, sp_field_type **f, char const name[], int flag);

int spSyncField(spMesh const *ctx, sp_field_type **f, int flag);


#endif /* SPFIELD_H_ */
