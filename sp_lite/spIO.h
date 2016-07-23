//
// Created by salmon on 16-7-23.
//

#ifndef SIMPLA_SPIO_H
#define SIMPLA_SPIO_H
#include "sp_lite_def.h"
#include <H5Ipublic.h>
#include <hdf5.h>

void spIOWriteSimple(spIOStream *os, const char *url,
                     int d_type,
                     void *d, int ndims,
                     size_type const *dims,
                     size_type const *start,
                     size_type const *stride,
                     size_type const *count,
                     size_type const *block,
                     int flag);

void spIOWriteIndexedBlockSimple(
    spIOStream *os,
    int num,
    const char **url,
    int d_type[],
    void **d,
    int num_of_block,
    size_type block_length,
    size_type const *offset,
    int flag)
{

};
#endif //SIMPLA_SPIO_H
