//
// Created by salmon on 16-9-13.
//

#ifndef SIMPLA_SPALOGORITHM_H
#define SIMPLA_SPALOGORITHM_H

#include "sp_lite_def.h"

int sort_by_key(size_type const *, size_type const *, size_type *);

/**
 *   dest[n]=src[index[n]]
 *   0<=n<num
 * @param dest
 * @param src
 * @param num
 * @param index
 * @return
 */
int spMemoryIndirectCopy(Real *dest, Real const *src, size_type num, size_type max_num, size_type const *index);

/**
 * v[n]=min+n
 * 0<= n < num
 * @param index
 * @param num
 * @param min
 * @return
 */
int spFillSeqInt(size_type *v, size_type num, size_type min, size_type step);

int spTransformMinus(size_type *v, size_type const *a, size_type const *b, size_type num);

int spTransformAdd(size_type *v, size_type const *a, size_type const *b, size_type num);

#endif //SIMPLA_SPALOGORITHM_H
