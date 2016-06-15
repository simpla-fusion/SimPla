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

void spCreateField(spMesh *ctx, sp_field_type **f, int iform);

void spDestroyField(spMesh *ctx, sp_field_type**f);

#endif /* SPFIELD_H_ */
