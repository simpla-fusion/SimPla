//
// Created by salmon on 16-6-7.
//

#include "../ParticleLite.h"

int main(int argc, char **argv)
{
    struct spElementDescription *p_desc = spParticleDescriptionCreate();
    struct spPageBuffer *p_buffer = spPageBufferCreate(p_desc);

    struct spPage *p = spPageCreate(p_buffer);

    spPageClose(p, p_buffer);
    spPageBufferClose(p_buffer);
    spParticleDescriptionClose(p_desc);
}