//
// Created by salmon on 16-6-9.
//

#ifndef SIMPLA_PARTICLEPAGE_H
#define SIMPLA_PARTICLEPAGE_H

#ifdef __cplusplus
extern "C" {

#include "SmallObjPool.h"

#endif

struct spPage;
struct spPagePool;


int spParticleCopy(size_t key, struct spPage const *src_page, struct spPage **dest_page, struct spPagePool *buffer);

int spParticleCopyN(size_t key, size_t s_num, struct spPage **s_page, struct spPage **dest_page, spPagePool *buffer);

void spParticleClear(size_t key, struct spPage **pg, struct spPagePool *buffer);


#ifdef __cplusplus
}
#endif
#endif //SIMPLA_PARTICLEPAGE_H
