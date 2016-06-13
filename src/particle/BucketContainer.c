//
// Created by salmon on 16-6-6.
//
#include "BucketContainer.h"

#include <memory.h>
#include <malloc.h>

#include <pthread.h>
#include "../../src/sp_config.h"

struct spPageGroup
{
    struct spPageGroup *next;
    size_type number_of_pages;
    struct spPage *m_pages;
    void *m_data;
};
struct spPagePool
{
    size_t ele_size_in_byte;
    struct spPageGroup *m_page_group_head;
    struct spPage *m_free_page;
    pthread_mutex_t m_pool_mutex_;

};
MC_DEVICE MC_HOST
struct spPagePool *spPagePoolCreate(size_t size_in_byte);

MC_DEVICE MC_HOST
void spPagePoolDestroy(struct spPagePool **pool);

MC_DEVICE MC_HOST
void spPagePoolRelease(struct spPagePool *pool);

MC_DEVICE MC_HOST
size_t spSizeInByte(struct spPagePool const *pool) { return pool->ele_size_in_byte; }

MC_DEVICE MC_HOST
struct spPageGroup *spPageGroupCreate(size_t ele_size_in_byte);


/****************************************************************************
 * Element access
 */
/**
 *  access the first element
 */
MC_DEVICE MC_HOST
struct spPage *spFront(struct spPage *p) { return p; }

/**
 *  access the last element
 */
MC_DEVICE MC_HOST
struct spPage *spBack(struct spPage *p)
{
    if (p != 0x0) { while (p->next != 0x0) { p = p->next; }}
    return p;
}


/****************************************************************************
 * Capacity
 */
MC_DEVICE MC_HOST
int spEmpty(struct spPage const *p)
{
    int res = 0;
    while (p != 0x0)
    {
        res += (p->flag > 0x0) ? 1 : 0;
        p = p->next;
    }
    return (res == 0) ? 1 : 0;
};
MC_DEVICE MC_HOST
int spFull(struct spPage const *p)
{
    int res = 0;
    while (p != 0x0)
    {
        res += ((p->flag + 1) != 0x0) ? 1 : 0;
        p = p->next;
    }
    return (res == 0) ? 1 : 0;
};
MC_DEVICE MC_HOST
size_t spMaxSize(struct spPage const *p)
{
    return (size_t) (-1);
};

MC_INLINE
size_t bit_count64(uint64_t x)
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

MC_DEVICE MC_HOST
size_t spSize(struct spPage const *p)
{
    size_t res = 0;
    while (p != 0x0)
    {
        res += bit_count64(p->flag);
        p = p->next;
    }
    return res;
}

MC_DEVICE MC_HOST
size_t spCapacity(struct spPage const *p)
{
    size_t res = 0;
    while (p != 0x0)
    {
        res += SP_NUMBER_OF_ELEMENT_IN_PAGE;
        p = p->next;
    }
    return res;
};

/****************************************************************************
 * Modifiers
 */
MC_DEVICE MC_HOST
size_t spMove(struct spPage **src, struct spPage **dest)
{
    size_t ret = 0;
    if (src != 0x0 && *src != 0x0)
    {
        struct spPage *tmp = *src;
        *src = (*src)->next;
        tmp->next = *dest;
        *dest = tmp;
        ret = 1;
    }
    return ret;
};
MC_DEVICE MC_HOST
void spMerge(struct spPage **src, struct spPage **dest)
{
    size_t count = 0;

    if (src != 0x0 && *src != 0x0)
    {
        if (*dest == 0x0)
        {
            *dest = *src;
        } else
        {
            spBack(*dest)->next = *src;
        }

        *src = 0x0;
    }

}

MC_DEVICE MC_HOST
size_t spMoveN(size_t n, struct spPage **src, struct spPage **dest)
{
    size_t count = 0;
    struct spPage *head = *src;
    while (*src != 0x0 && count < n)
    {
        *src = (*src)->next;
        ++count;
    }
    struct spPage *tail = *src;
    *src = (*src)->next;
    tail->next = *dest;
    *dest = tail;

    return count;
};

//void spReserve(struct spPage **p, size_t num, struct spPagePool *pool)
//{
//    if (num > spCapacity(*p))
//    {
//        struct spPage *t = spPageCreateN(pool, (size_t) ((num - spCapacity(*p)) / SP_NUMBER_OF_ELEMENT_IN_PAGE + 1));
//        spPushFront(p, &t);
//    }
//
//
//};

MC_DEVICE MC_HOST
void spSetPageFlag(struct spPage *p, size_t tag)
{
    while (p != 0x0)
    {
        p->flag = tag;
        p = p->next;
    }
}

MC_DEVICE MC_HOST
void spClear(struct spPage **p, struct spPage **buffer)
{
    while (*p != 0x0)
    {
        if ((*p)->flag == 0x0) { spMove(p, buffer); }
        else { *p = (*p)->next; }
    }
};


//size_t spInsert(struct spPage *p, size_t N, size_t size_in_byte, void const *src)
//{
//    void *dest = 0x0;
//    for (struct spOutputIterator __it = {0x0, 0x0, p, size_in_byte};
//         (N > 0) && ((dest = spItInsert(&__it)) != 0x0); --N)
//    {
//        memcpy(dest, src, size_in_byte);
//        src += size_in_byte;
//    }
//
//    return N;
//}

MC_DEVICE MC_HOST
size_t spFill(struct spPage *p, size_t N, size_t size_in_byte, const byte_type *src)
{
    while (N > 0 && p != 0x0)
    {
        size_t n = (N < SP_NUMBER_OF_ELEMENT_IN_PAGE) ? N : SP_NUMBER_OF_ELEMENT_IN_PAGE;

        memcpy(p->data, src, size_in_byte * n);

        src += size_in_byte * n;

        N -= n;

        p = p->next;
    }
    return N;

}

MC_DEVICE MC_HOST
void *spNext(struct spOutputIterator *it)
{
    //TODO need optimize
    void *ret = 0x0;
    for (; (*(it->page)) != 0x0; (*(it->page)) = (*(it->page))->next)
    {
        if (it->flag == 0x0 || it->p == 0x0)
        {
            it->flag = 0x01;
            it->p = (*(it->page))->data;
        }
        else
        {
            it->flag <<= 1;
            it->p += it->ele_size_in_byte;
        }
        while ((((*(it->page))->flag & (it->flag)) == 0) && (it->flag != 0))
        {
            it->flag <<= 1;
            it->p += it->ele_size_in_byte;
        }
        if (it->flag != 0)
        {
            ret = it->p;
            break;
        }
    }
    return ret;
}

MC_DEVICE MC_HOST
void *spItTraverse(struct spOutputIterator *it)
{
    //TODO need optimize
    void *ret = 0x0;
//    for (; (*(it->page)) != 0x0; (*(it->page)) = (*(it->page))->next)
//    {
//        if (it->tag == 0x0 || it->p == 0x0)
//        {
//            it->tag = 0x01;
//            it->p = (*(it->page))->data;
//        }
//        else
//        {
//            it->tag <<= 1;
//            it->p += it->ele_size_in_byte;
//        }
//        while ((((*(it->page))->tag & (it->tag)) == 0) && (it->tag != 0))
//        {
//            it->tag <<= 1;
//            it->p += it->ele_size_in_byte;
//        }
//        if (it->tag != 0)
//        {
//            ret = it->p;
//            break;
//        }
//    }
    return ret;
}

MC_DEVICE MC_HOST
void *spInputIteratorNext(struct spInputIterator *it)
{
    if (it == 0x0) { return 0x0; }

    struct spPage **pg = (it->page);

    while (1)
    {
        if (*pg == 0x0) { (*pg) = spPageCreate(it->pool, 1); }

        if (it->tag == 0x0 || it->p == 0x0)
        {
            it->tag = 0x1;
            it->p = (*pg)->data;
        }


        it->tag <<= 1;
        it->p += it->pool->ele_size_in_byte;

        if ((*pg)->flag + 1 == 0x0) { pg = &((*pg)->next); }
        if (((*pg)->flag & (it->tag)) == 0x0) { break; }
    }

    (*pg)->flag |= it->tag;


    return it->p;
}

MC_DEVICE MC_HOST
size_t spNextBlank2(struct spPage **pg, size_t *tag, byte_type **p, struct spPagePool *pool)
{
    START:
    if (*pg == 0x0) { *pg = spPageCreate(pool, 1); }

    if (tag == 0x0 || *p == 0x0)
    {
        *tag = 0x1;
        *p = (*pg)->data;
    }


    while (((*pg)->flag & (*tag)) != 0)
    {
        (*tag) <<= 1;
        *p += pool->ele_size_in_byte;

        if ((*pg)->flag + 1 == 0x0 || (*tag) == 0x0)
        {
            pg = &((*pg)->next);
            goto START;
        }
    }

    (*pg)->flag |= (*tag);
    return (*tag);
}

MC_DEVICE MC_HOST
void *spItInsert(struct spOutputIterator *it)
{
    //TODO need optimize
    void *ret = 0x0;
//    if ((*(it->page)) == 0x0) { goto RETURN; }
//
//    if (it->tag == 0x0 || it->p == 0x0)
//    {
//        it->tag = 0x1;
//        it->p = (*(it->page))->data;
//    }
//
//
//    while (((*(it->page))->tag & it->tag) != 0)
//    {
//        it->tag <<= 1;
//        it->p += it->ele_size_in_byte;
//
//        if ((*(it->page))->tag + 1 == 0x0 || it->tag == 0x0)
//        {
//            if ((*(it->page))->next == 0x0) { goto RETURN; }
//            (*(it->page)) = (*(it->page))->next;
//            it->tag = 0x01;
//            it->p = (*(it->page))->data;
//        }
//    }
//    (*(it->page))->tag |= it->tag;
//    ret = it->p;
//    RETURN:
    return ret;
}

MC_DEVICE MC_HOST
void *spItRemoveIf(struct spOutputIterator *it, int flag)
{
    //TODO need optimize
    void *ret = 0x0;
//    for (; (*(it->page)) != 0x0; (*(it->page)) = (*(it->page))->next)
//    {
//
//        if (it->tag == 0x0 || it->p == 0x0)
//        {
//            it->tag = 0x01;
//            it->p = (*(it->page))->data;
//        }
//        else
//        {
//            if (flag > 0) { (*(it->page))->tag &= ~(it->tag); }
//
//            it->tag <<= 1;
//            it->p += it->ele_size_in_byte;
//        }
//        while ((((*(it->page))->tag & (it->tag)) == 0) && (it->tag != 0))
//        {
//            it->tag <<= 1;
//            it->p += it->ele_size_in_byte;
//        }
//        if (it->tag != 0)
//        {
//            ret = it->p;
//            break;
//        }
//    }
    return ret;
}

/*******************************************************************************************/
MC_DEVICE MC_HOST
struct spPage *spPageCreate(struct spPagePool *pool, size_t num)
{
    pthread_mutex_lock(&(pool->m_pool_mutex_));
    struct spPage *head = 0x0;
    struct spPage **tail = &(pool->m_free_page);
    while (num > 0)
    {
        if ((*tail) == 0x0)
        {
            struct spPageGroup *pg = spPageGroupCreate(pool->ele_size_in_byte);
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
MC_DEVICE MC_HOST
size_t spPageDestroyN(struct spPagePool *pool, struct spPage **p, size_t num)
{
    pthread_mutex_lock(&(pool->m_pool_mutex_));
    size_t res = spMoveN(num, p, &(pool->m_free_page));
    pthread_mutex_unlock(&(pool->m_pool_mutex_));

    return res;
}

MC_DEVICE MC_HOST
void spPageDestroy(struct spPagePool *pool, struct spPage **p)
{
    pthread_mutex_lock(&(pool->m_pool_mutex_));
    spMerge(p, &(pool->m_free_page));
    pthread_mutex_unlock(&(pool->m_pool_mutex_));

}

MC_DEVICE MC_HOST
/*******************************************************************************************/
struct spPageGroup *spPageGroupCreate(size_t ele_size_in_byte)
{
    struct spPageGroup *ret = (struct spPageGroup *) (malloc(sizeof(struct spPageGroup)));
    ret->next = 0x0;
    ret->m_data = malloc(ele_size_in_byte
                         * SP_NUMBER_OF_ELEMENT_IN_PAGE
                         * SP_NUMBER_OF_PAGES_IN_GROUP);
    for (int i = 0; i < SP_NUMBER_OF_PAGES_IN_GROUP; ++i)
    {
        ret->m_pages[i].next = &(ret->m_pages[i + 1]);
        ret->m_pages[i].data = ret->m_data + i * (ele_size_in_byte
                                                  * SP_NUMBER_OF_ELEMENT_IN_PAGE);
    }
    ret->m_pages[SP_NUMBER_OF_PAGES_IN_GROUP - 1].next = 0x0;
    return ret;
}

/**
 * @return next page group
 */
MC_DEVICE MC_HOST
void spPageGroupDestroy(struct spPageGroup **pg)
{
    free((*pg)->m_data);
    free((*pg));
    (*pg) = 0x0;
}

/**
 *  @return first free page
 *    pg = first page group
 */
MC_DEVICE MC_HOST
struct spPage *spPageGroupClean(struct spPageGroup **pg)
{
    struct spPage *ret = 0x0;
    while (*pg != 0x0)
    {
        struct spPageGroup *pg0 = *pg;
        (*pg) = (*pg)->next;
        struct spPage *tmp = ret;
        size_t count = 0;
        for (int i = 0; i < SP_NUMBER_OF_PAGES_IN_GROUP; ++i)
        {
            if (pg0->m_pages[i].flag == 0)
            {
                pg0->m_pages[i].next = ret;
                ret = &(pg0->m_pages[i]);
                ++count;
            }
        }
        //
        if (count >= SP_NUMBER_OF_PAGES_IN_GROUP) //group is empty
        {
            ret = tmp;
            spPageGroupDestroy(&pg0);
        }

    }

    return ret;
}

MC_DEVICE MC_HOST
struct spPagePool *spPagePoolCreate(size_t size_in_byte)
{
    struct spPagePool *res = (struct spPagePool *) (malloc(sizeof(struct spPagePool)));
    res->m_page_group_head = 0x0;
    res->m_free_page = 0x0;
    res->ele_size_in_byte = size_in_byte;
    pthread_mutex_init(&(res->m_pool_mutex_), NULL);
    return res;
}

MC_DEVICE MC_HOST
void spPagePoolDestroy(struct spPagePool **pool)
{

    while ((*pool)->m_page_group_head != 0x0)
    {
        struct spPageGroup *pg = (*pool)->m_page_group_head;
        (*pool)->m_page_group_head = (*pool)->m_page_group_head->next;
        spPageGroupDestroy(&pg);
    }
    free(*pool);
    (*pool) = 0x0;

}

MC_DEVICE MC_HOST
void spPagePoolRelease(struct spPagePool *pool)
{

}


MC_GLOBAL
void spBucketResortKernel(struct spPage **buckets, int ndims, size_type const *dims, struct spPagePool *pool)
{

    MC_SHARED struct spPage *read_buffer[CACHE_SIZE];
    MC_SHARED struct spPage *write_buffer[CACHE_SIZE];
    MC_SHARED
    status_flag_type shift_flag[CACHE_SIZE];

    size_type ele_size_in_byte = args->ele_size_in_type;
#ifdef __CUDACC__
    for (size_type _blk_s = blockIdx.x, _blk_e = args->number_of_idx; _blk_s < _blk_e; _blk_s += blockDim.x)
#else
    for (size_type _blk_s = 0, _blk_e = args->number_of_idx; _blk_s < _blk_e; ++_blk_s)
#endif
    {
        size_type cell_idx = args->cell_idx[_blk_s];

        // read tE,tB from E,B
        // clear tJ
        MC_SHARED  struct spPage **self;

        // TODO load data to cache

        for (int n = 0; n < CACHE_SIZE; ++n)
        {
            struct spPage const *from = read_buffer[n];

            status_flag_type p_flag = shift_flag[n];

            while (from != 0x0)
            {
                byte_type const *v = from->data;

#ifdef __CUDACC__
                int i = threadIdx.x;
#else
                for (int i = 0; i < SP_NUMBER_OF_ELEMENT_IN_PAGE; ++i)
#endif
                {
                    status_flag_type flag = 0x1UL << i;

                    struct boris_point_s *p = (struct boris_point_s *) (v + ele_size_in_byte * i);

                    if ((from->flag & flag) != 0 && ((p->_tag & 0x3F) == p_flag))
                    {


                        struct boris_point_s *p_next = (struct boris_point_s *) spAtomicInsert(self,
                                                                                               args->pool);
                        p_next->_tag = p->_tag;

//                        spBorisPushOne(p, p_next, args->cmr, dt, E, B, args->inv_dx);

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


void spBucketResort(struct spPage **buckets, int ndims, size_type const *dims, struct spPagePool *pool)
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
