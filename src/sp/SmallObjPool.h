//
// Created by salmon on 16-6-6.
//

#ifndef SIMPLA_SMALLOBJPOOL_H_
#define SIMPLA_SMALLOBJPOOL_H_

#ifdef __cplusplus
namespace simpla { namespace sp
{
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

struct spPage *spPageCreateN(struct spPagePool *pool, size_t num);

void spPageClose(struct spPage **p, struct spPagePool *pool);

struct spPagePool *spPagePoolCreate(size_t size_in_byte);

void spPagePoolClose(struct spPagePool **pool);

/****************************************************************************
 * Element access
 */
/**
 *  access the first element
 */
struct spPage *spFront(struct spPage *p);

/**
 *  access the last element
 */
struct spPage *spBack(struct spPage *p);

/****************************************************************************
 * Capacity
 */

/**
 *  checks whether the container is empty
 *  @return if empty return 1, else return 0
 */
int spEmpty(struct spPage const *p);


/**
 *  checks whether the container is full
 *  @return if full return 1, else return 0
 */
int spFull(struct spPage const *p);

/**
 * @return  the number of elements
 */
size_t spSize(struct spPage const *p);

/**
 * @ returns the maximum possible number of elements
 */
size_t spMaxSize(struct spPage const *p);

/**
 * @return the number of elements that can be held in currently allocated storage
 */
size_t spCapacity(struct spPage const *p);

/**
 *  reserves storage
 */
void spReserve(struct spPage **p, size_t num, struct spPagePool *pool);


/**
 * reduces memory usage by freeing unused memory
 * @return removed page
 */
void spShrinkToFit(struct spPage **p, struct spPagePool *);

/****************************************************************************
 * Modifiers
 */

/**
 * 	clears the contents, but do not release memory
 * 	size=0, but do not change capacity
 *
 */
void spClear(struct spPage *);

/**
 * 	remove
 * 	@return *p=(old *p)->next;
 */
void spRemove(struct spPage **p, struct spPagePool *pool);

/** adds an element to the end
 *  @return last element
 **/
void spPushBack(struct spPage **, struct spPage *);

/** removes the last element
 *  @return last element
 **/
void spPopBack(struct spPage **, struct spPagePool *);

void spPushFront(struct spPage **, struct spPage **);

void spPopFront(struct spPage **, struct spPagePool *);

/**
 *  1. insert 'num' objects to 'dest' page
 *  @return if 'p' is full then return number of remain objects;
 *          else return 0;
 */
size_t spInsert(struct spPage *p, size_t N, size_t size_in_byte, void const *src);


size_t spFill(struct spPage *p, size_t N, size_t size_in_byte, void const *src);
/****************************************************************************
 * Operations
 */
/**
 * merges two sorted lists
 */
void spMerge(struct spPage *, struct spPage **);


/****************************************************************************
 * Iterators
 */

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
void *spItTraverse(struct spIterator *it);

/**
 * 1. insert new obj to page
 * @return if page is full return 0x0, else return pointer to new obj
 */
void *spItInsert(struct spIterator *it);


/**
 * 1. if flag > 0 then remove 'current obj', else do nothing
 * 2. move 'it' to next obj
 * @return if 'next obj' is available then return pointer to 'next obj',
 *          else return 0x0
 */
void *spItRemoveIf(struct spIterator *it, int flag);

#ifndef __cplusplus
/**
 *   traverses all element
 *   example:
 *       SP_FOREACH_ELEMENT(struct point_s, p, pg){ p->x = 0;}
 */
#define SP_OBJ_FOREACH(__TYPE__, __PTR__, __PG_HEAD__)          \
__TYPE__ *__PTR__ = 0x0; \
for (struct spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(__PTR__ =  spItTraverse(&__it)) != 0x0;)

/**
 * insert elements to page .
 * example:
 * SP_ADD_NEW_ELEMENT(200,struct point_s, p, pg, p_pool) {p->x = 0; }
 */
#define SP_OBJ_INSERT(__NUMBER__, __TYPE__, __PTR__, __PG_HEAD__)          \
__TYPE__ *__PTR__; size_t __count = __NUMBER__; \
for (struct spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(__PTR__ =  spItInsert(&__it )) != 0x0 && (__count>1);--__count)

#define SP_OBJ_REMOVE_IF(__TYPE__, __PTR__, __PG_HEAD__, __TAG__)          \
__TYPE__ *__PTR__ = 0x0; int __TAG__=0;\
for (struct spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(__PTR__ =  spItRemoveIf(&__it,__TAG__)) != 0x0;)

#else
}// extern "C" {

/**
 *   traverses all element
 *   example:
 *       SP_FOREACH_ELEMENT(struct point_s, p, pg){ p->x = 0;}
 */
#define SP_OBJ_FOREACH(__TYPE__, __PTR__, __PG_HEAD__)          \
__TYPE__ *__PTR__ = 0x0; \
for (sp::spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(__PTR__ =  reinterpret_cast<__TYPE__ *>(sp::spItTraverse(&__it))) != 0x0;)

/**
 * insert elements to page .
 * example:
 * SP_ADD_NEW_ELEMENT(200,struct point_s, p, pg) {p->x = 0; }
 */
#define SP_OBJ_INSERT(__NUMBER__, __TYPE__, __PTR__, __PG_HEAD__)          \
__TYPE__ *__PTR__; size_t __count = __NUMBER__; \
for (sp::spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(__PTR__ =  reinterpret_cast<__TYPE__ *>(sp::spItInsert(&__it))) != 0x0 && (__count>1);--__count)

#define SP_OBJ_REMOVE_IF(__TYPE__, __PTR__, __PG_HEAD__, __TAG__)          \
__TYPE__ *__PTR__ = 0x0; int __TAG__=0;\
for (sp::spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(__PTR__ = reinterpret_cast<__TYPE__ *>( sp::spItRemoveIf(&__it,__TAG__))) != 0x0;)


std::shared_ptr<spPagePool> makePagePool(size_t size_in_byte)
{
    return std::shared_ptr<spPagePool>(spPagePoolCreate(size_in_byte), [=](spPagePool *pg) { spPagePoolClose(&pg ); });
}

std::shared_ptr<spPage> makePage(std::shared_ptr<spPagePool> pool)
{
    return std::shared_ptr<spPage>(spPageCreate(pool.get()), [=](spPage *pg) { spPageClose(&pg, pool.get()); });
}


}} //namespace simpla { namespace sp

#endif
#endif //SIMPLA_SMALLOBJPOOL_H_
