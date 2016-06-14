//
// Created by salmon on 16-6-9.
//

#ifndef SIMPLA_PARTICLEPAGE_H
#define SIMPLA_PARTICLEPAGE_H

#ifdef __cplusplus
extern "C" {

#include "BucketContainer.h"

#endif

int spParticleCopy(size_t key,  spPage const *src_page,  spPage **dest_page,  spPagePool *buffer);

int spParticleCopyN(size_t key, size_t s_num,  spPage **s_page,  spPage **dest_page, spPagePool *buffer);

void spParticleClear(size_t key,  spPage **pg,  spPagePool *buffer);

#ifdef __cplusplus
}
#endif
#endif //SIMPLA_PARTICLEPAGE_H
