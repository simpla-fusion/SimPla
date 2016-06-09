//
// Created by salmon on 16-6-9.
//

#ifndef SIMPLA_PARTICLEPAGE_H
#define SIMPLA_PARTICLEPAGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "../sp/SmallObjPool.h"

#ifdef __cplusplus
}
namespace simpla { namespace sp
{
extern "C" {
#endif

struct spPage;

#ifdef __cplusplus
}
}}//namespace simpla { namespace sp

namespace simpla { namespace particle
{
extern "C" {
#endif

int spParticleCopy(size_t key, size_t size_in_byte, struct spPage const *src_page, struct spPage **dest_page,
                   struct spPage **buffer);

int spParticleCopyN(size_t key, size_t size_in_byte, size_t src_num, struct spPage const *src_page[],
                    struct spPage **dest_page, struct spPage **buffer);

void spParticleClean(size_t key, size_t size_in_byte, struct spPage **pg, struct spPage **buffer);

#ifdef __cplusplus
}
}}//namespace simpla { namespace particle
#endif
#endif //SIMPLA_PARTICLEPAGE_H
