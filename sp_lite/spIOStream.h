//
// Created by salmon on 16-9-12.
//

#ifndef SIMPLA_SPIOSTREAM_H
#define SIMPLA_SPIOSTREAM_H


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

int spIOStreamWriteSimple(spIOStream *,
                          const char *name,
                          struct spDataType_s const *d_type,
                          void *d,
                          int ndims,
                          int const *dims,
                          int const *start,
                          int const *stride,
                          int const *count,
                          int const *block,
                          int const *g_dims,
                          int const *g_start,
                          int flag);

#endif //SIMPLA_SPIOSTREAM_H
