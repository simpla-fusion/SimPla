//
// Created by salmon on 16-6-8.
//

#ifndef SIMPLA_PARTICLECOMMON_H
#define SIMPLA_PARTICLECOMMON_H


#ifdef __cplusplus

#include <cstddef>

namespace simpla { namespace particle
{
extern "C" {

#else
#   include <stddef.h>
#endif

#define POINT_HEAD size_t _cell; size_t _tag;

struct point_head
{
    size_t _cell;
    size_t _tag;
    char data[];
};

#ifdef __cplusplus
}// extern "C" {
}}//namespace simpla { namespace particle
#endif

#endif //SIMPLA_PARTICLECOMMON_H
