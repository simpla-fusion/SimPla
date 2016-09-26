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
int spMemoryCopyIndirect(Real *dest, Real const *src, size_type num, size_type const *index);

int spMemoryCopyInvIndirect(Real *dest, Real const *src, size_type num, size_type const *index);


int spMemoryCopySubArray(void *dest, void const *src, int type_tag, size_type const *strides, size_type const *start,
                         size_type const *count);

int spMemoryCopyInvSubArray(void *dest, void const *src, int type_tag, size_type const *strides, size_type const *start,
                            size_type const *count);

/**
 * v[n]=min+n
 * 0<= n < num
 * @param index
 * @param num
 * @param min
 * @return
 */
int spFillSeq(void *v, int type_tag, size_type num, size_type min, size_type step);

int spTransformMinus(size_type *v, size_type const *a, size_type const *b, size_type num);

int spTransformAdd(size_type *v, size_type const *a, size_type const *b, size_type num);


int spPackInt(size_type **dest, int *num, size_type const *src,
              size_type num_of_cell, size_type const *start, size_type const *count);


#endif //SIMPLA_SPALOGORITHM_H
