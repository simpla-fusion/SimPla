/*
 * spIOStream.h
 *
 *  Created on: 2016年7月6日
 *      Author: salmon
 */

#ifndef SPIOSTREAM_H_
#define SPIOSTREAM_H_
#include "sp_def.h"
#include "spDataModel.h"
#ifdef __cplusplus
extern "C"
{
#endif

enum
{
	SP_FILE_NEW = 1UL << 1, SP_FILE_APPEND = 1UL << 2, SP_FILE_BUFFER = (1UL << 3), SP_FILE_RECORD = (1UL << 4)
};

struct spIOStream_s;
typedef struct spIOStream_s spIOStream;

int spIOStreamCreate(struct spIOStream_s **);

int spIOStreamDestroy(struct spIOStream_s **);

int spIOStreamOpen(struct spIOStream_s *, char const * url, int flag);

int spIOStreamClose(struct spIOStream_s *);

int spIOStreamWrite(struct spIOStream_s *, char const * name, int flag, spDataSet const *);

int spIOStreamRead(struct spIOStream_s *, char const * name, spDataSet ** data);

#ifdef __cplusplus
} //extern "C"
#endif

#endif /* SPIOSTREAM_H_ */
