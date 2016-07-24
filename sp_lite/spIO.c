//
// Created by salmon on 16-7-23.
//

#include "spIO.h"

void spIOWriteSimple(spIOStream *os,
                     const char *url,
                     struct spDataType_s const *d_type,
                     void *d,
                     int ndims,
                     size_type const *dims,
                     size_type const *start,
                     size_type const *stride,
                     size_type const *count,
                     size_type const *block,
                     int flag)
{

    spIOStreamWriteSimple(os, url,
                          d_type,
                          d,
                          ndims,
                          dims,
                          start,
                          stride,
                          count,
                          block,
                          flag
    );
}

