//
// Created by salmon on 16-9-13.
//
#include "../../spAlogorithm.h"

int sort_by_key(size_type const *key_start, size_type const *key_end, size_type *value)
{
    UNIMPLEMENTED;
    return SP_DO_NOTHING;
};

int spMemoryRelativeCopy(Real *dest, Real const *src, size_type num, size_type const *index)
{
    for (int i = 0; i < num; ++i) { dest[i] = src[index[i]]; }
    return SP_SUCCESS;
}
