//
// Created by salmon on 16-9-12.
//

#ifndef SIMPLA_SPIOSTREAM_H
#define SIMPLA_SPIOSTREAM_H
#include "sp_lite_config.h"
enum
{
    SP_FILE_NEW = 1UL << 1, SP_FILE_APPEND = 1UL << 2, SP_FILE_BUFFER = (1UL << 3), SP_FILE_RECORD = (1UL << 4)
};
struct spIOStream_s;

typedef struct spIOStream_s spIOStream;

int spIOStreamCreate(spIOStream **);

int spIOStreamDestroy(spIOStream **);

int spIOStreamPWD(spIOStream *, char *url);

int spIOStreamOpen(spIOStream *, const char *url);

int spIOStreamClose(spIOStream *);

//int spIOStreamWrite(spIOStream *, const char *name, spDataSet const *, int tag);
//
//int spIOStreamRead(spIOStream *, const char *name, spDataSet const *, int tag);

struct spDataType_s;

int spIOStreamWriteSimple(spIOStream *,
                          const char *name,
                          int data_type_tag,
                          void *d,
                          int ndims,
                          size_type const *dims,
                          size_type const *start,
                          size_type const *stride,
                          size_type const *count,
                          size_type const *block,
                          size_type const *g_dims,
                          size_type const *g_start,
                          int flag);
#endif //SIMPLA_SPIOSTREAM_H
