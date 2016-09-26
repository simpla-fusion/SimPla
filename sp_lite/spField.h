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

int spFieldCreate(spField **f, const struct spMesh_s *m, int iform, int type_tag);

int spFieldDestroy(spField **f);

int spFieldDeploy(spField *f);

int spFieldDataType(spField const *f);

void *spFieldData(spField *f);

size_type spFieldGetSizeInByte(spField const *f);

int spFieldIsSoA(spField const *f);

void *spFieldDeviceData(spField *f);

const void *spFieldDataConst(spField const *f);

const void *spFieldDeviceDataConst(spField const *f);

int spFieldClear(spField *f);

int spFieldShow(const spField *f, char const *name);

#define SHOW_FIELD(_F_)  {printf( "%s:%d:0:%s: Display field [ %s ]", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_F_) );SP_CALL(spFieldShow(_F_,NULL));}

int spFieldWrite(spField *f, struct spIOStream_s *os, char const name[], int flag);

int spFieldRead(spField *f, struct spIOStream_s *os, char const name[], int flag);

int spFieldSync(spField *f);

int spFieldNumberOfSub(spField const *f);

int spFieldSubArray(spField *f, void **data);

int spFieldCopyToHost(void **d, spField const *f);

int spFieldCopyToDevice(spField *f, void const *d);

/* device dependent function*/
int spFieldFill(spField *f, int tag, void const *v);

int spFieldFillIntSeq(spField *f, int tag, size_type min, size_type step);

int spFieldAddScalar(spField *f, void const *);

int spFieldMinusScalar(spField *f, void const *);

int spFieldMultiplyScalar(spField *f, void const *);

int spFieldDivideScalar(spField *f, void const *);

int spFieldAssign(spField *f, spField const *);

int spFieldAdd(spField *f, spField const *);

int spFieldMinus(spField *f, spField const *);

int spFieldMultiply(spField *f, spField const *);

int spFieldDivide(spField *f, spField const *);


int spFieldFillSeq(spField *f, int domain_tag);

struct spMesh_s;
typedef struct spMesh_s spMesh;

int spFieldTestSync(spMesh const *m, int type_tag);


#endif /* SPFIELD_H_ */
