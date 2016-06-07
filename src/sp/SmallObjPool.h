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


#define SP_NUMBER_OF_ELEMENT_IN_PAGE 64
typedef uint64_t status_tag_type;
struct spPage
{
    size_t obj_size_in_byte;
    status_tag_type tag;
    void *data;
    struct spPage *next;
};
struct spPagePool;

struct spPage *spPageCreate(struct spPagePool *pool);

void spPageClose(struct spPage *p, struct spPagePool *pool);

size_t spPageCount(struct spPage const *p);

struct spPagePool *spPagePoolCreate(size_t size_in_byte);

void spPagePoolClose(struct spPagePool *pool);

void spInsertObj(size_t num, void const *src, struct spPage *dest, struct spPagePool *pool);

struct spIterator
{
    status_tag_type tag;
    void *p;
    struct spPage *page;
    size_t obj_size_in_byte;
};

/**
 * 1. move 'it' to 'next obj'
 *
 * @return  if 'next obj' is available then return pointer to 'next obj'
 *           else return 0x0
 */
void *spTraverseIterator(struct spIterator *it);

/**
 * 1. insert new obj to page
 * @return pointer to new obj
 */
void *spInsertIterator(struct spIterator *it, struct spPagePool *pool);

/**
 * 1. if flag > 0 then remove 'current obj', else do nothing
 * 2. move 'it' to next obj
 * @return if 'next obj' is available then return pointer to 'next obj',
 *          else return 0x0
 */
void *spRemoveIfIterator(struct spIterator *it, int flag);

/**
 *   traverses all element
 *   example:
 *       SP_FOREACH_ELEMENT(struct point_s, p, pg){ p->x = 0;}
 */
#define SP_ELEMENT_FOREACH(__TYPE__, __PTR__, __PG_HEAD__)          \
__TYPE__ *__PTR__ = 0x0; \
for (struct spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(__PTR__ = spTraverseIterator(&__it)) != 0x0;)

/**
 * insert elements to page .
 * example:
 * SP_ADD_NEW_ELEMENT(200,struct point_s, p, pg, p_pool) {p->x = 0; }
 */
#define SP_OBJ_INSERT(__NUMBER__, __TYPE__, __PTR__, __PG_HEAD__, __POOL__)          \
__TYPE__ *__PTR__; size_t __count = __NUMBER__; \
for (struct spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(__PTR__ = spInsertIterator(&__it, __POOL__)) != 0x0 && (__count>1);--__count)

#define SP_ELEMENT_REMOVE_IF(__TYPE__, __PTR__, __PG_HEAD__, __TAG__)          \
__TYPE__ *__PTR__ = 0x0; int __TAG__=0;\
for (struct spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(__PTR__ = spRemoveIfIterator(&__it,__TAG__)) != 0x0;)
#ifdef __cplusplus
}
#endif
#endif //SIMPLA_PARTICLELITE_H
