/*
 * spField.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPFIELD_H_
#define SPFIELD_H_
#include "sp_lite_def.h"


struct spMesh_s;

struct spDataType_s;
struct spIOStream_s;
struct spField_s;

typedef struct spField_s spField;

int spFieldCreate(spField **f, const struct spMesh_s *m, int iform, int type_tag);

int spFieldDestroy(spField **f);

int spFieldDeploy(spField *f);

struct spDataType_s const *spFieldDataType(spField const *f);

void *spFieldData(spField *f);
int spFieldUseSoA(spField const *f);

void *spFieldDeviceData(spField *f);

const void *spFieldDataConst(spField const *f);

const void *spFieldDeviceDataConst(spField const *f);

int spFieldClear(spField *f);

int spFieldFill(spField *f, Real v);

int spFieldWrite(spField *f, struct spIOStream_s *os, char const name[], int flag);

int spFieldRead(spField *f, struct spIOStream_s *os, char const name[], int flag);

int spFieldSync(spField *f);

#endif /* SPFIELD_H_ */
