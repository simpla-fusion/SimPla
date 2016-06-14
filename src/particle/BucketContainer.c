//
// Created by salmon on 16-6-6.
//
#include "BucketContainer.h"

#include <memory.h>
#include <malloc.h>

#include <pthread.h>
#include "../../src/sp_config.h"

typedef struct spPageGroup_s
{
    struct spPageGroup_s *next;
    size_type number_of_pages;
    spPage *m_pages;
    void *m_data;
} spPageGroup;

typedef struct spPagePool_s
{
    size_type entity_size_in_byte;
    spPageGroup *m_page_group_head;
    spPage *m_free_page;
    pthread_mutex_t m_pool_mutex_;

} spPagePool;

/*******************************************************************************************/



MC_DEVICE MC_HOST
size_type spPagePoolEntitySizeInByte(spPagePool const *pool) { return pool->entity_size_in_byte; }


MC_DEVICE MC_HOST
spPageGroup *spPageGroupCreate(size_type entity_size_in_byte, size_type num_of_pages)
{
    spPageGroup *res = 0x0;

#ifdef __CUDACC__
    res = (spPageGroup *) (cudaMalloc(sizeof(spPageGroup)));
    res->m_pages = cudaMalloc(sizeof(spPage) * num_of_pages);
    res->m_data = cudaMalloc(entity_size_in_byte * SP_NUMBER_OF_ELEMENT_IN_PAGE * num_of_pages);
#else
    res = (spPageGroup *) (malloc(sizeof(spPageGroup)));
    res->m_pages = malloc(sizeof(spPage) * num_of_pages);
    res->m_data = malloc(entity_size_in_byte * SP_NUMBER_OF_ELEMENT_IN_PAGE * num_of_pages);
#endif

    res->next = 0x0;
    res->number_of_pages = num_of_pages;

    for (int i = 0; i < num_of_pages; ++i)
    {
        res->m_pages[i].next = &(res->m_pages[i + 1]);
        res->m_pages[i].flag = 0x0;
        res->m_pages[i].tag = 0x0;
        res->m_pages[i].entity_size_in_byte = entity_size_in_byte;
        res->m_pages[i].data = res->m_data + i * (entity_size_in_byte
                                                  * SP_NUMBER_OF_ELEMENT_IN_PAGE);
    }
    res->m_pages[num_of_pages - 1].next = 0x0;
    return res;
}

/**
 * @return next page group
 */
MC_DEVICE MC_HOST
void spPageGroupDestroy(spPageGroup **pg)
{
    if (pg != 0 && *pg != 0)
    {
        spPageGroup *t = (*pg);
        (*pg) = (*pg)->next;

#ifdef __CUDACC__
        cudaFree(t->m_data);
        cudaFree(t->m_pages);
        cudaFree(t);
#else
        free((*pg)->m_data);
        free((*pg)->m_pages);
        free((*pg));
#endif

    }
}

/**
 *  @return first free page
 *    pg = first page group
 */
MC_DEVICE MC_HOST
size_type spPageGroupSize(spPageGroup const *pg)
{

    size_type count = 0;

    for (int i = 0; i < pg->number_of_pages; ++i)
    {
        count += spPageNumberOfEntities(&(pg->m_pages[i]));
    }
    return count;
}

MC_DEVICE MC_HOST
spPagePool *spPagePoolCreate(size_type size_in_byte)
{
    spPagePool *res = 0x0;

#ifdef __CUDACC__
    res = (spPagePool *) (cudaMalloc(sizeof(spPagePool)));
#else
    res = (spPagePool *) (malloc(sizeof(spPagePool)));
#endif
    res->entity_size_in_byte = size_in_byte;
    res->m_page_group_head = 0x0;
    res->m_free_page = 0x0;
#ifndef __CUDACC__
    pthread_mutex_init(&(res->m_pool_mutex_), NULL);
#endif
    return res;
}

MC_DEVICE MC_HOST
void spPagePoolDestroy(spPagePool **pool)
{

    while ((*pool)->m_page_group_head != 0x0)
    {
        spPageGroup *pg = (*pool)->m_page_group_head;
        (*pool)->m_page_group_head = (*pool)->m_page_group_head->next;
        spPageGroupDestroy(&pg);
    }
    pthread_mutex_destroy(&(*pool)->m_pool_mutex_);
    free(*pool);
    (*pool) = 0x0;

}

MC_DEVICE MC_HOST
void spPagePoolReleaseEnpty(spPagePool *pool)
{
    spPageGroup *head = pool->m_page_group_head;
    while (head != 0x0)
    {
        if (spPageGroupSize(head) == 0)
        {
            spPageGroupDestroy(&head);
        }
        else
        {
            head = head->next;
        }
    }
}



/****************************************************************************
 *  Page create and modify
 */

MC_DEVICE MC_HOST
spPage *spPageCreate(size_type num, spPagePool *pool)
{
    pthread_mutex_lock(&(pool->m_pool_mutex_));
    spPage *head = 0x0;
    spPage **tail = &(pool->m_free_page);
    while (num > 0)
    {
        if ((*tail) == 0x0)
        {
            spPageGroup *pg = spPageGroupCreate(pool->entity_size_in_byte);
            pg->next = pool->m_page_group_head;
            pool->m_page_group_head = pg;
            (*tail) = &((pool->m_page_group_head->m_pages)[0]);
        }
        if (head == 0x0)
        {
            head = (*tail);
        }

        while (num > 0 && (*tail) != 0x0)
        {
            tail = &((*tail)->next);
            --num;
        }

    }
    pool->m_free_page = (*tail)->next;
    (*tail)->next = 0x0;
    pthread_mutex_unlock(&(pool->m_pool_mutex_));
    return head;
};


MC_DEVICE MC_HOST size_t
spPageDestroy(spPage **p, spPagePool *pool)
{
    pthread_mutex_lock(&(pool->m_pool_mutex_));

    size_type res = spPageSize(*p);

    spPagePushFront(&(pool->m_free_page), p);

    pthread_mutex_unlock(&(pool->m_pool_mutex_));

    return res;
}


MC_DEVICE MC_HOST void
spPagePushFront(spPage **p, spPage *f)
{
    if (f != 0x0)
    {
        *spPageBack(&f) = *p;
        *p = f;
    }
}

MC_DEVICE MC_HOST spPage *
spPagePopFront(spPage **p)
{
    spPage *res = 0x0;
    if (p != 0x0 && *p != 0x0)
    {
        res = *p;
        *p = (*p)->next;

        res->next = 0x0;
    }
    return res;
};

MC_DEVICE MC_HOST size_t
spPageSplice(spPage **self, spPage **other)
{
    spPagePushFront(self, spPagePopFront(other));
};



/****************************************************************************
 * Element access
 */

MC_DEVICE MC_HOST spPage **spPageFront(spPage **p)
{
    return p;
};


MC_DEVICE MC_HOST spPage **spPageBack(spPage **p)
{
    while (p != 0x0 && *p != 0x0 && (*p)->next != 0x0) { p = &((*p)->next); }
    return p;
}



/****************************************************************************
 * Capacity
 */
MC_DEVICE MC_HOST size_t
spPageSize(spPage const *p)
{
    size_type res = 0;
    while (p != 0x0)
    {
        ++res;
        p = p->next;
    }
    return res;
}


MC_DEVICE MC_HOST int
spPageIsEmpty(spPage const *p)
{
    int count = 0;
    while (p != 0x0)
    {
        count += (p->flag != 0x0) ? 1 : 0;
        p = p->next;
    }

    return (count > 0) ? 0 : 1;
};
MC_DEVICE MC_HOST int
spPageIsFull(spPage const *p)
{
    if (p == 0x0) { return 0; }
    else
    {
        int count = 0;
        while (p != 0x0)
        {
            count += ((p->flag + 1) != 0x0) ? 1 : 0;
            p = p->next;
        }
        return count;
    }
};


MC_INLINE size_type
bit_count64(uint64_t x)
{
    static const uint64_t m1 = 0x5555555555555555; //binary: 0101...
    static const uint64_t m2 = 0x3333333333333333; //binary: 00110011..
    static const uint64_t m4 = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
    static const uint64_t m8 = 0x00ff00ff00ff00ff; //binary:  8 zeros,  8 ones ...
    static const uint64_t m16 = 0x0000ffff0000ffff; //binary: 16 zeros, 16 ones ...
    static const uint64_t m32 = 0x00000000ffffffff; //binary: 32 zeros, 32 ones


    x = (x & m1) + ((x >> 1) & m1); //put count of each  2 bits into those  2 bits
    x = (x & m2) + ((x >> 2) & m2); //put count of each  4 bits into those  4 bits
    x = (x & m4) + ((x >> 4) & m4); //put count of each  8 bits into those  8 bits
    x = (x & m8) + ((x >> 8) & m8); //put count of each 16 bits into those 16 bits
    x = (x & m16) + ((x >> 16) & m16); //put count of each 32 bits into those 32 bits
    x = (x & m32) + ((x >> 32) & m32); //put count of each 64 bits into those 64 bits
    return (size_t) x;

}

MC_DEVICE MC_HOST size_type
spPageNumberOfEntities(spPage const *p)
{
    size_type res = 0;
    while (p != 0x0)
    {
        res += bit_count64(p->flag);
        p = p->next;
    }
    return res;
}

MC_DEVICE MC_HOST size_type
spPageCapacity(spPage const *p)
{
    return spPageSize(p) * SP_NUMBER_OF_ELEMENT_IN_PAGE;
}

/***************************************************************************/
/*  Entity
 **/


MC_DEVICE MC_HOST void
spEntityClear(spPage *p)
{
    while (p != 0x0)
    {
        p->flag = 0x0;
        p = p->next;
    }
};

MC_DEVICE MC_HOST
size_type spEntityFill(spPage *p, size_type num, const byte_type *src)
{
    while (num > 0 && p != 0x0)
    {
        size_type n = (num < SP_NUMBER_OF_ELEMENT_IN_PAGE) ? num : SP_NUMBER_OF_ELEMENT_IN_PAGE;

        memcpy(p->data, src, p->entity_size_in_byte * n);

        src += p->entity_size_in_byte * n;

        num -= n;

        p->flag = (bucket_page_status_flag_t) (0 - 1);
        p = p->next;
    }
    return num;

}

MC_DEVICE MC_HOST spEntity *
spEntityInsert(spPage *pg)
{
    spPage *t = pg;
    bucket_page_status_flag_t flag = 0x0;
    return spEntityInsertWithHint(&t, &flag);
}

MC_DEVICE MC_HOST spEntity *
spEntityInsertWithHint(spPage **pg, bucket_page_status_flag_t *flag)
{
    byte_type *res = 0x0;
    if (*flag == 0x0) { *flag = 0x1; }

    while ((*pg) != 0)
    {
        res = (*pg)->data;

        while (((*pg)->flag + 1 != 0x0) && *flag != 0x0)
        {
            if (((*pg)->flag & *flag) == 0x0)
            {
                (*pg)->flag |= *flag;
                goto RETURN;
            }

            res += (*pg)->entity_size_in_byte;
            *flag <<= 1;

        }

        *flag = 0x1;
        pg = &(*pg)->next;

    }
    RETURN:
    return (spEntity *) res;
}

MC_DEVICE MC_HOST spEntity *
spEntityNext(spPage **pg, bucket_page_status_flag_t *flag)
{

    byte_type *res = 0x0;
    if (*flag == 0x0) { *flag = 0x1; }

    while ((*pg) != 0)
    {
        res = (*pg)->data;

        while (((*pg)->flag != 0x0) && *flag != 0x0)
        {
            if (((*pg)->flag & *flag) != 0x0) { goto RETURN; }

            res += (*pg)->entity_size_in_byte;
            *flag <<= 1;

        }

        *flag = 0x1;
        pg = &(*pg)->next;

    }
    RETURN:
    return (spEntity *) res;
}

MC_DEVICE MC_HOST void spEntityRemove(spPage *p, bucket_page_status_flag_t flag)
{
    p->flag &= (~flag);
};


MC_GLOBAL
void spBucketResortKernel(spPage **buckets, size_type number_of_idx, size_type const *idx, int ndims,
                          size_type const *dims, spPagePool *pool)
{
    size_type NUM_OF_NEIGHBOUR = 27;

    MC_SHARED  spPage *read_buffer[NUM_OF_NEIGHBOUR];
    MC_SHARED  spPage *write_buffer[NUM_OF_NEIGHBOUR];
    MC_SHARED
    bucket_page_status_flag_t shift_flag[NUM_OF_NEIGHBOUR];

    size_type ele_size_in_byte = spPagePoolEntitySizeInByte(pool);
#ifdef __CUDACC__
    for (size_type _blk_s = blockIdx.x, _blk_e = args->number_of_idx; _blk_s < _blk_e; _blk_s += blockDim.x)
#else
    for (size_type _blk_s = 0, _blk_e = number_of_idx; _blk_s < _blk_e; ++_blk_s)
#endif
    {
        size_type cell_idx = idx[_blk_s];

        // read tE,tB from E,B
        // clear tJ
        MC_SHARED   spPage **self;

        // TODO load data to cache

        for (int n = 0; n < NUM_OF_NEIGHBOUR; ++n)
        {
            spPage const *from = read_buffer[n];

            bucket_page_status_flag_t p_flag = shift_flag[n];

            while (from != 0x0)
            {
                byte_type const *v = from->data;

#ifdef __CUDACC__
                int i = threadIdx.x;
#else
                for (int i = 0; i < SP_NUMBER_OF_ELEMENT_IN_PAGE; ++i)
#endif
                {
                    bucket_page_status_flag_t flag = 0x1UL << i;

                    boris_point_s *p = (boris_point_s * )(v + ele_size_in_byte * i);

                    if ((from->flag & flag) != 0 && ((p->_tag & 0x3F) == p_flag))
                    {


                        spEntity *p_next = spAtomicInsert(self, pool);
                        p_next->_tag = p->_tag;

                        COPY(p, p_next);

                        p_next->_tag &= ~(0x3F);
                        // TODO r -> tag
//                        p_next->_tag |= spParticleFlag(p_next->r);


                    }
                }
                from = from->next;

            }//   while (from != 0x0)

        }// foreach CACHE
#ifdef __CUDACC__
        __syncthreads();
    MC_COPY(self, &(next[cell_idx]));
#else
        spPushFront(self, &(buckets[cell_idx]));


#endif
    }//foreach block

}


void spBucketResort(spPage **buckets, int ndims, size_type const *dims, spPagePool *pool)
{
#ifdef __CUDACC__
    /* @formatter:off*/
     int numBlocks(number_of_core/SP_NUMBER_OF_ELEMENT_IN_PAGE);
     dim3 threadsPerBlock(SP_NUMBER_OF_ELEMENT_IN_PAGE, 1);
     spBucketResortKernel<<<numBlocks,threadsPerBlock >>> (args,  dt,prev,next, fE, fB,fRho,fJ);
/* @formatter:on*/
#else
    spBucketResortKernel(buckets, ndims, dims, pool);
#endif
}
