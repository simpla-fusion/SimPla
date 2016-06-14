//
// Created by salmon on 16-6-12.
//

#include "ParticleUtility.h"

size_t spInsertParticle(struct spPage **pg, size_t N, const byte_type *src,
                        struct spPagePool *pool)
{
    while (*pg != 0x0)
    {

        pg = &((*pg)->next);
    }
    bucket_page_status_flag_t flag = 0x1;
    size_type ele_size_in_byte = spPagePoolEntitySizeInByte(pool);
    size_type num_of_pages = N / SP_NUMBER_OF_ELEMENT_IN_PAGE;
    struct spPage *head = spPageCreate(num_of_pages, pool);
    struct spPage *tail = head;
    while (N > SP_NUMBER_OF_ELEMENT_IN_PAGE)
    {


        struct spPage *head = spPageCreate(num_of_pages, pool);
        struct spPage *tail = head;
        while ()

    }
    if (*pg == 0x0) { *pg = spPageCreate(N / SP_NUMBER_OF_ELEMENT_IN_PAGE + 1, pool); };

    while (*pg != 0x0)
    {
        byte_type *p = (*pg)->data;

        while ((*pg)->flag & flag != 0)
        {
            flag <<= 1;
            src += ele_size_in_byte;
        }
    }

    if (flag == 0x0)
    {
        flag = 0x1;
        *pg = (*pg)->next;
    }


};
