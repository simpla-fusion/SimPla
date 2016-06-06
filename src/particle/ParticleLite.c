//
// Created by salmon on 16-6-6.
//

#include <memory.h>
#include "ParticleLite.h"

struct spElementDescription *spElementDescriptionCreate()
{

}

void spElementDescriptionClose(struct spElementDescription *dest)
{

}

void spElementDescriptionCopy(struct spElementDescription const *src,
                              struct spElementDescription *dest)
{
    dest->size_in_byte = src->size_in_byte;
}

struct spPage *spPageCreate(struct spPageBuffer *buffer)
{
    if (buffer->m_free_page == 0x0) { spPageBufferClose(buffer); }
    struct spPage *ret = buffer->m_free_page;
    buffer->m_free_page = buffer->m_free_page->next;
    return ret;
}

void spPageClose(struct spPage *p, struct spPageBuffer *buffer)
{
    p->next = buffer->m_free_page;
    buffer->m_free_page = p;
}

void spPageAdd(struct spPage *p, struct spPage **head)
{
    p->next = *head;
    *head = p;
}

size_t spPageCount(struct spPage const *p) { return bit_count(p->tag); }

struct spPageGroup *spPageGroupCreate(struct spElementDescription const *m_ele_desc)
{
    struct spPageGroup *ret = (struct spPageGroup *) (malloc(sizeof(struct spPageGroup)));
    ret->next = 0x0;
    ret->m_data = malloc(m_ele_desc->size_in_byte
                         * SP_NUMBER_OF_ELEMENT_IN_PAGE
                         * SP_NUMBER_OF_PAGES_IN_GROUP);
    for (int i = 0; i < SP_NUMBER_OF_PAGES_IN_GROUP; ++i)
    {
        ret->m_pages[i].next = &(ret->m_pages[i + 1]);
        ret->m_pages[i].data = ret->m_data + i * (m_ele_desc->size_in_byte
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
        count += spPageCount(&pg->m_pages[i]);
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
    while (pg != 0x0)
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

struct spPageBuffer *spPageBufferCreate(struct spElementDescription const *m_element_desc)
{
    struct spPageBuffer *res = (struct spPageBuffer *) (malloc(sizeof(struct spPageBuffer)));
    res->head = 0x0;
    spElementDescriptionCopy(m_element_desc, &res->m_element_desc);

    return res;
}


void spPageBufferExtent(struct spPageBuffer *buffer)
{
    struct spPageGroup *pg = spPageGroupCreate(&(buffer->m_element_desc));
    pg->next = buffer->head;
    buffer->head = pg;
    pg->m_pages[SP_NUMBER_OF_PAGES_IN_GROUP - 1].next = buffer->m_free_page;
    buffer->m_free_page = &(pg->m_pages[0]);
}

void spPageBufferClose(struct spPageBuffer *buffer)
{
    while (buffer->head != 0x0)
    {
        buffer->head = spPageGroupClose(buffer->head);
    }

}


void spAddElements(size_t num, void const *src, struct spPage **head, struct spPageBuffer *buffer)
{
    if (num == 0) { return; }
    if ((*head)->tag + 1 == 0UL)
    {
        struct spPage *tmp = spPageCreate(buffer);
        tmp->next = (*head);
        (*head) = tmp;
    }
    void *dest = (*head)->data;

    for (size_t tag = 0x1; tag != 0 && num > 0; tag <<= 1)
    {
        if (((*head)->tag & tag) == 0)
        {
            strncpy(dest, src, buffer->m_element_desc.size_in_byte);
            src += buffer->m_element_desc.size_in_byte;
            --num;
            (*head)->tag |= tag;
        }
        dest += buffer->m_element_desc.size_in_byte;
    }

    spAddElements(num, src, head, buffer);

}
