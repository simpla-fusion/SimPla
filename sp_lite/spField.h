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

int spFieldDataType(spField const *f);

void *spFieldData(spField_t f);

size_type spFieldGetSizeInByte(spField const *f);

int spFieldIsSoA(spField const *f);

void *spFieldDeviceData(spField_t f);

const void *spFieldDataConst(spField const *f);

const void *spFieldDeviceDataConst(spField const *f);

int spFieldClear(spField_t f);

int spFieldWrite(spField_t f, struct spIOStream_s *os, char const name[], int flag);

int spFieldRead(spField_t f, struct spIOStream_s *os, char const name[], int flag);

int spFieldSync(spField_t f);

int spFieldNumberOfSub(spField const *f);

int spFieldSubArray(spField_t f, void **data);

int spFieldCopyToHost(void **d, spField const *f);

int spFieldCopyToDevice(spField *f, void const *d);

/* device dependent function*/
int spFieldFill(spField *f, int tag, void const *v);
int spFieldFillIntSeq(spField_t f, int tag, size_type min, size_type step);

int spFieldAddScalar(spField *f, void const *);
int spFieldMinusScalar(spField *f, void const *);
int spFieldMultiplyScalar(spField *f, void const *);
int spFieldDivideScalar(spField *f, void const *);

int spFieldAssign(spField *f, spField const *);
int spFieldAdd(spField *f, spField const *);
int spFieldMinus(spField *f, spField const *);
int spFieldMultiply(spField *f, spField const *);
int spFieldDivide(spField *f, spField const *);


#endif /* SPFIELD_H_ */
