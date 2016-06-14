//
// Created by salmon on 16-6-9.
//
#include <memory.h>
#include "BucketContainer.h"
#include "ParticleInterface.h"


int spParticleCopy(size_t key, struct spPage const *src_page, struct spPage **dest_page, struct spPagePool *pool)
{

    size_t size_in_byte = spPagePoolEntitySizeInByte(pool);

    size_t src_tag = 0x0, dest_tag = 0x0;

    void *src_v = 0x0, *dest_v = 0x0;
//
//    while (src_page != 0x0)
//    {
//        if (src_tag == 0x0)
//        {
//            src_tag = 0x1;
//            src_v = src_page->data;
//        }
//
//        if (((src_page->tag & src_tag) != 0) && (((struct point_head *) (src_v))->_cell == key))
//        {
//            if (dest_tag == 0x0)
//            {
//                if (*dest_page == 0x0)
//                {
//                    *dest_page = *buffer;
//                    if (*buffer == 0x0) { return SP_BUFFER_EMPTY; }
//                    *buffer = (*buffer)->next;
//                    (*dest_page)->next = 0x0;
//                }
//                dest_v = (*dest_page)->data;
//                dest_tag = 0x1;
//            }
//
//            memcpy(dest_v, src_v, size_in_byte);
//
//            dest_tag <<= 1;
//            dest_v += size_in_byte;
//            if (dest_tag == 0x0) { (*dest_page) = (*dest_page)->next; }
//
//        }
//
//        src_tag <<= 1;
//        src_v += size_in_byte;
//        if (src_tag == 0x0) { src_page = src_page->next; }
//    }
    return SP_SUCCESS;
}

int spParticleCopyN(size_t key, size_t src_num, struct spPage **src_page, struct spPage **dest_page,
                    struct spPagePool *buffer)
{
    for (int i = 0; i < src_num; ++i)
    {
        spParticleCopy(key, src_page[i], dest_page, buffer);
    }
    return src_num;
}

void spParticleClear(size_t key, struct spPage **pg, struct spPagePool *buffer)
{
    size_t size_in_byte = spPagePoolEntitySizeInByte(buffer);

    if (*pg == 0x0) { return; }

    size_t src_tag = 0x0, dest_tag = 0x0;

    void *src_v = 0x0, *dest_v = 0x0;

    struct spPage *trash = 0x0;

    while ((*pg) != 0x0)
    {
        if (src_tag == 0x0)
        {
            src_tag = 0x1;
            src_v = (*pg)->data;
        }

//        if ((((struct point_head *) (src_v))->_cell != key))
        {
            (*pg)->flag &= ~src_tag;
        }

        src_tag <<= 1;
        src_v += size_in_byte;
        if ((*pg)->flag == 0x0) { spPageMove(pg, &trash); }
        else if (src_tag == 0x0) { (*pg) = (*pg)->next; }

    }

    spPageDestroy(buffer, NULL);
}
