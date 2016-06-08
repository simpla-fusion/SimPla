//
// Created by salmon on 16-6-6.
//

#include <memory.h>
#include <malloc.h>

#ifndef __STDC_NO_THREADS__
#include <threads.h>
#endif

#include "SmallObjPool.h"

#define SP_NUMBER_OF_PAGES_IN_GROUP 64

struct spPageGroup
{
    struct spPageGroup *next;
    struct spPage m_pages[SP_NUMBER_OF_PAGES_IN_GROUP];
    status_tag_type tag;
    void *m_data;
};
struct spPagePool
{
#ifndef __STDC_NO_THREADS__
    mtx_t m_mutex_;
#endif

    size_t obj_size_in_byte;
    struct spPageGroup *head;
    struct spPage *m_free_page;
};


struct spPage *spPageCreate(struct spPagePool *pool)
{

    if (pool->m_free_page == 0x0) { spPagePoolExtent(pool); }
    struct spPage *ret = pool->m_free_page;
    pool->m_free_page = pool->m_free_page->next;
    ret->next = 0x0;
    ret->tag = 0x0;

    return ret;
}

struct spPage *spPageCreateN(struct spPagePool *pool, size_t num)
{
    struct spPage *front = spPageCreate(pool);
    struct spPage *back = front;
    for (int i = 0; i < num - 1; ++i)
    {
        back->next = spPageCreate(pool);
        back = back->next;
    }
    return front;
};

void spPageClose(struct spPage **p, struct spPagePool *pool)
{
    spClear(*p);
    spPushFront(&(pool->m_free_page), *p);
    *p = 0x0;
}


struct spPageGroup *spPageGroupCreate(size_t obj_size_in_byte)
{
    struct spPageGroup *ret = (struct spPageGroup *) (malloc(sizeof(struct spPageGroup)));
    ret->next = 0x0;
    ret->m_data = malloc(obj_size_in_byte
                         * SP_NUMBER_OF_ELEMENT_IN_PAGE
                         * SP_NUMBER_OF_PAGES_IN_GROUP);
    for (int i = 0; i < SP_NUMBER_OF_PAGES_IN_GROUP; ++i)
    {
        ret->m_pages[i].next = &(ret->m_pages[i + 1]);
        ret->m_pages[i].data = ret->m_data + i * (obj_size_in_byte
                                                  * SP_NUMBER_OF_ELEMENT_IN_PAGE);
    }
    ret->m_pages[SP_NUMBER_OF_PAGES_IN_GROUP - 1].next = 0x0;
    return ret;
}

/**
 * @return next page group
 */
struct spPageGroup *spPageGroupClose(struct spPageGroup *pg)
{
    struct spPageGroup *ret;
    ret = pg->next;
    free(pg->m_data);
    free(pg);
    return ret;
}

size_t spPageGroupCount(struct spPageGroup const *pg)
{
    size_t count = 0;
    for (int i = 0; i < SP_NUMBER_OF_PAGES_IN_GROUP; ++i)
    {
        count += spSize(&pg->m_pages[i]);
    }
    return count;
}

/**
 *  @return first free page
 *    pg = first page group
 */
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
            if (pg0->m_pages[i].tag == 0)
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
            spPageGroupClose(pg0);
        }

    }

    return ret;
}

struct spPagePool *spPagePoolCreate(size_t size_in_byte)
{
    struct spPagePool *res = (struct spPagePool *) (malloc(sizeof(struct spPagePool)));
    res->head = 0x0;
    res->m_free_page = 0x0;
    res->obj_size_in_byte = size_in_byte;
    return res;
}


void spPagePoolExtent(struct spPagePool *pool)
{
    struct spPageGroup *pg = spPageGroupCreate(pool->obj_size_in_byte);
    pg->next = pool->head;
    pool->head = pg;
    pg->m_pages[SP_NUMBER_OF_PAGES_IN_GROUP - 1].next = pool->m_free_page;
    pool->m_free_page = &(pg->m_pages[0]);

}

void spPagePoolClose(struct spPagePool **pool)
{

    while ((*pool)->head != 0x0)
    {
        (*pool)->head = spPageGroupClose((*pool)->head);
    }
    (*pool) = 0x0;

}
/****************************************************************************
 * Element access
 */
/**
 *  access the first element
 */
struct spPage *spFront(struct spPage *p) { return p; }

/**
 *  access the last element
 */
struct spPage *spBack(struct spPage *p)
{
    if (p != 0x0) { while (p->next != 0x0) { p = p->next; }}
    return p;
}


/****************************************************************************
 * Capacity
 */

int spEmpty(struct spPage const *p)
{
    int res = 0;
    while (p != 0x0)
    {
        res += (p->tag > 0x0) ? 1 : 0;
        p = p->next;
    }
    return (res == 0) ? 1 : 0;
};

int spFull(struct spPage const *p)
{
    int res = 0;
    while (p != 0x0)
    {
        res += ((p->tag + 1) != 0x0) ? 1 : 0;
        p = p->next;
    }
    return (res == 0) ? 1 : 0;
};

size_t spMaxSize(struct spPage const *p)
{
    return (size_t) (-1);
};

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

size_t spSize(struct spPage const *p)
{
    size_t res = 0;
    while (p != 0x0)
    {
        res += bit_count64(p->tag);
        p = p->next;
    }
    return res;
}

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

void spReserve(struct spPage **p, size_t num, struct spPagePool *pool)
{
    if (num > spCapacity(*p))
    {
        spPushFront(p, spPageCreateN(pool, (size_t) ((num - spCapacity(*p)) / SP_NUMBER_OF_ELEMENT_IN_PAGE + 1)));
    }


};

void spShrinkToFit(struct spPage **p, struct spPagePool *pool)
{
    if (p == 0x0) { return; }

    struct spPage **t = p;
    while (*t != 0x0)
    {
        if ((*t)->tag == 0x0) { spRemove(t, pool); }
        else { t = &((*t)->next); }
    }

};

/** adds an element to the end */
void spPushBack(struct spPage **p, struct spPage *other)
{
    if (p == 0x0) { return; }
    else
    {
        while ((*p) != 0x0) { p = &((*p)->next); }
        *p = other;
    }


};

/** removes the last element */
void spPopBack(struct spPage **p, struct spPagePool *pool)
{
    if (p != 0x0 && *p != 0x0)
    {
        while ((*p)->next != 0x0) { p = &((*p)->next); }
        spRemove(p, pool);
    }

}

void spPushFront(struct spPage **p, struct spPage *other)
{
    if (other != 0x0)
    {
        other->next = *p;
        *p = other;
    }

};

void spPopFront(struct spPage **p, struct spPagePool *pool)
{
    spRemove(p, pool);
};

/****************************************************************************
 * Modifiers
 */

void spClear(struct spPage *p)
{
    while (p != 0x0)
    {
        p->tag = 0x0;
        p = p->next;
    }
};

void spRemove(struct spPage **p, struct spPagePool *pool)
{
    if (p != 0x0 && *p != 0x0)
    {
        struct spPage *t = *p;
        *p = (*p)->next;
        t->next = 0x0;
        spPushFront(&(pool->m_free_page), t);
    }
}


size_t spInsert(struct spPage *p, size_t N, size_t size_in_byte, void const *src)
{
    void *dest = 0x0;
    for (struct spIterator __it = {0x0, 0x0, p, size_in_byte};
         (N > 0) && ((dest = spItInsert(&__it)) != 0x0); --N)
    {
        memcpy(dest, src, size_in_byte);
        src += size_in_byte;
    }

    return N;
}


void *spItTraverse(struct spIterator *it)
{
    //TODO need optimize
    void *ret = 0x0;
    for (; it->page != 0x0; it->page = it->page->next)
    {
        if (it->tag == 0x0 || it->p == 0x0)
        {
            it->tag = 0x01;
            it->p = it->page->data;
        }
        else
        {
            it->tag <<= 1;
            it->p += it->obj_size_in_byte;
        }
        while (((it->page->tag & (it->tag)) == 0) && (it->tag != 0))
        {
            it->tag <<= 1;
            it->p += it->obj_size_in_byte;
        }
        if (it->tag != 0)
        {
            ret = it->p;
            break;
        }
    }
    return ret;
}

void *spItInsert(struct spIterator *it)
{
    //TODO need optimize
    void *ret = 0x0;
    if (it->page == 0x0) { goto RETURN; }

    if (it->tag == 0x0 || it->p == 0x0)
    {
        it->tag = 0x1;
        it->p = it->page->data;
    }


    while ((it->page->tag & it->tag) != 0)
    {
        it->tag <<= 1;
        it->p += it->obj_size_in_byte;

        if (it->page->tag + 1 == 0x0 || it->tag == 0x0)
        {
            if (it->page->next == 0x0) { goto RETURN; }
            it->page = it->page->next;
            it->tag = 0x01;
            it->p = it->page->data;
        }
    }
    it->page->tag |= it->tag;
    ret = it->p;
    RETURN:
    return ret;
}


void *spItRemoveIf(struct spIterator *it, int flag)
{
    //TODO need optimize
    void *ret = 0x0;
    for (; it->page != 0x0; it->page = it->page->next)
    {

        if (it->tag == 0x0 || it->p == 0x0)
        {
            it->tag = 0x01;
            it->p = it->page->data;
        }
        else
        {
            if (flag > 0) { it->page->tag &= ~(it->tag); }

            it->tag <<= 1;
            it->p += it->obj_size_in_byte;
        }
        while (((it->page->tag & (it->tag)) == 0) && (it->tag != 0))
        {
            it->tag <<= 1;
            it->p += it->obj_size_in_byte;
        }
        if (it->tag != 0)
        {
            ret = it->p;
            break;
        }
    }
    return ret;
}


