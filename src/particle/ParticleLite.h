//
// Created by salmon on 16-6-6.
//

#ifndef SIMPLA_PARTICLELITE_H
#define SIMPLA_PARTICLELITE_H
#ifdef __cplusplus
extern "C" {
#endif


#include <stddef.h>
#include <stdint.h>
#include <malloc.h>
#include <hdf5.h>

#define   BIT_NUMBER_OF_TAG   (sizeof(sp_tag_t) * 8)


/**
 * Bit Count: Parallel Counting – MIT HAKMEM
 */
inline size_t bit_count(uint64_t u)
{
    uint64_t uCount;
    uCount = u
             - ((u >> 1) & 033333333333)
             - ((u >> 2) & 011111111111);
    return (size_t) (((uCount + (uCount >> 3)) & 030707070707) % 63);
}

typedef uint64_t sp_tag_t;

struct spElementDescription
{
    size_t size_in_byte;
};

struct spElementDescription *spElementDescriptionCreate();

void spElementDescriptionClose(struct spElementDescription *dest);

void spElementDescriptionCopy(struct spElementDescription const *src,
                              struct spElementDescription *dest);


#define SP_NUMBER_OF_PAGES_IN_GROUP 64
#define SP_NUMBER_OF_ELEMENT_IN_PAGE 64
typedef uint64_t status_tag_type;
struct spPage
{
    struct spPage *next;
    struct spElementDescription const *type;
    status_tag_type tag;
    void *data;
};
struct spPageGroup
{
    struct spPageGroup *next;
    struct spPage m_pages[SP_NUMBER_OF_PAGES_IN_GROUP];
    status_tag_type tag;
    void *m_data;
};
struct spPageBuffer
{
    struct spElementDescription m_element_desc;
    struct spPageGroup *head;
    struct spPage *m_free_page;
};

struct spPage *spPageCreate(struct spPageBuffer *buffer);

void spPageClose(struct spPage *p, struct spPageBuffer *buffer);

void spPageAdd(struct spPage *p, struct spPage **head);

struct spPageGroup *spPageGroupCreate(struct spElementDescription const *m_ele_desc);

struct spPageGroup *spPageGroupClose(struct spPageGroup *pg);

size_t spPageGroupCount(struct spPageGroup const *pg);

struct spPage *spPageGroupClean(struct spPageGroup **pg);

struct spPageBuffer *spPageBufferCreate(struct spElementDescription const *m_element_desc);

void spPageBufferExtent(struct spPageBuffer *);

void spPageBufferClose(struct spPageBuffer *res);

void spAddElements(size_t num, void const *src,
                   struct spPage **head, struct spPageBuffer *buffer);

//void add(size_t num, element_s const *src, spPage **page);
//
//void add(element_s const *p, spPage *page);
//
//void push(element_s *p, double dt, field_array E, field_array B);
//
//void push_page(spPage *head, double dt, field_array E, field_array B);

#ifdef __cplusplus
}
#endif
#endif //SIMPLA_PARTICLELITE_H
