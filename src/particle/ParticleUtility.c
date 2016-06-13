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
    status_flag_type flag = 0x1;
    size_type ele_size_in_byte = spSizeInByte(pool);
    size_type num_of_pages = N / SP_NUMBER_OF_ELEMENT_IN_PAGE;
    struct spPage *head = spPageCreate(pool, num_of_pages);
    struct spPage *tail = head;
    while (N > SP_NUMBER_OF_ELEMENT_IN_PAGE)
    {


        struct spPage *head = spPageCreate(pool, num_of_pages);
        struct spPage *tail = head;
        while ()

    }
    if (*pg == 0x0) { *pg = spPageCreate(pool, N / SP_NUMBER_OF_ELEMENT_IN_PAGE + 1); };

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
