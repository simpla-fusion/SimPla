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
typedef struct spField_s *spField_t;
typedef const struct spField_s *spField_const_t;

int spFieldCreate(spField_t *f, const struct spMesh_s *m, int iform, int type_tag);

int spFieldDestroy(spField_t *f);

int spFieldDeploy(spField_t f);

struct spDataType_s const *spFieldDataType(spField const *f);

void *spFieldData(spField_t f);

size_type spFieldGetSizeInByte(spField const *f);

int spFieldIsSoA(spField const *f);

void *spFieldDeviceData(spField_t f);

const void *spFieldDataConst(spField const *f);

const void *spFieldDeviceDataConst(spField const *f);

int spFieldClear(spField_t f);

int spFieldFill(spField_t f, Real v);

int spFieldWrite(spField_t f, struct spIOStream_s *os, char const name[], int flag);

int spFieldRead(spField_t f, struct spIOStream_s *os, char const name[], int flag);

int spFieldSync(spField_t f);

int spFieldNumberOfSub(spField const *f);

int spFieldSubArray(spField_t f, void **data);

int spFieldAssign(spField_t f, int num, size_type const *offset, Real const **v);

int spFieldAdd(spField *f, void const *);

int spFieldMultify(spField *f, void const *);

int spFieldCopyToHost(void **d, spField const *f);

int spFieldCopyToDevice(spField *f, void const *d);

#endif /* SPFIELD_H_ */
