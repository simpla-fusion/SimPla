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

enum { SP_SUCCESS, SP_BUFFER_EMPTY };

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

struct spPagePool *spPagePoolCreate(size_t size_in_byte);

void spPagePoolDestroy(struct spPagePool **pool);

/**
 * release empty page group
 * @NOTE not complete
 */
void spPagePoolRelease(struct spPagePool *pool);

/****************************************************************************
 *  Create
 */
/**
 *  pop 'num' pages from pool
 *  @return pointer of first page
 */
struct spPage *spPageCreate(struct spPagePool *pool, size_t num);

/**
 * push 'num' pages back to pool
 * @return number of actually destroyed pages
 *         *p point to  the first remained page,
 *         if no page is remained *p=0x0
 */
size_t spPageDestroy(struct spPagePool *pool, struct spPage **p, size_t num);



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



/****************************************************************************
 * Modifiers
 */
/**
 *  move first page from "src" to "dest"
 *  @return number of moved page, if success return 1 else return 0
 */
size_t spMove(struct spPage **src, struct spPage **dest);

size_t spMoveN(size_t n, struct spPage **src, struct spPage **dest);

/**
 * 	Merge 'src' to 'dest'
 */
void spMerge(struct spPage **src, struct spPage **dest);

/**
 * move empty page to buffer
 */
void spClear(struct spPage **p, struct spPage **buffer);

/**
* set all page->tag=0x0
*/
void spSetTag(struct spPage *p, size_t tag);
//
//void spPushFront(struct spPage **, struct spPage **);
//
//void spPopFront(struct spPage **, struct spPage **);
//
///** adds an element to the end
// *  @return last element
// **/
//void spPushBack(struct spPage **, struct spPage *);
//
///** removes the last element
// *  @return last element
// **/
//void spPopBack(struct spPage **, struct spPage **buffer);

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


/****************************************************************************
 * Iterators
 */

struct spIterator
{
    status_tag_type tag;
    void *p;
    struct spPage *page;
    size_t ele_size_in_byte;
};

/**
 * 1. move 'it' to 'next obj'
 *
 * @return  if 'next obj' is available then return pointer to 'next obj'
 *           else return 0x0
 */
void *spNext(struct spIterator *it);

/**
 * 1. insert new obj to page
 * @return if page is full return 0x0, else return pointer to new obj
 */
void *spNextBlank(struct spIterator *it);


/**
 * 1. if flag > 0 then remove 'current obj', else do nothing
 * 2. move 'it' to next obj
 * @return if 'next obj' is available then return pointer to 'next obj',
 *          else return 0x0
 */
void *spItRemoveIf(struct spIterator *it, int flag);


struct spPagePool;

struct spPagePool *spPagePoolCreate(size_t size_in_byte);

void spPagePoolDestroy(struct spPagePool **pool);


#ifndef __cplusplus
/**
 *   traverses all element
 *   example:
 *       SP_FOREACH_ELEMENT(struct point_s, p, pg){ p->x = 0;}
 */
#define SP_PAGE_FOREACH(__TYPE__, _PTR_NAME_, __PG_HEAD__)          \
__TYPE__ *_PTR_NAME_ = 0x0; \
for (struct spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(_PTR_NAME_ =  spNext(&__it)) != 0x0;)

/**
 * insert elements to page .
 * example:
 * SP_ADD_NEW_ELEMENT(200,struct point_s, p, pg, p_pool) {p->x = 0; }
 */
#define SP_PAGE_INSERT(__NUMBER__, __TYPE__, _PTR_NAME_, __PG_HEAD__)          \
__TYPE__ *_PTR_NAME_; size_t __count = __NUMBER__; \
for (struct spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(_PTR_NAME_ =  spNextBlank(&__it )) != 0x0 && (__count>1);--__count)

#define SP_OBJ_REMOVE_IF(__TYPE__, _PTR_NAME_, __PG_HEAD__, __TAG__)          \
__TYPE__ *_PTR_NAME_ = 0x0; int __TAG__=0;\
for (struct spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(_PTR_NAME_ =  spItRemoveIf(&__it,__TAG__)) != 0x0;)

#else
}// extern "C" {

/**
 *   traverses all element
 *   example:
 *       SP_FOREACH_ELEMENT(struct point_s, p, pg){ p->x = 0;}
 */
#define SP_PAGE_FOREACH(__TYPE__, _PTR_NAME_, __PG_HEAD__)          \
__TYPE__ *_PTR_NAME_ = 0x0; \
for (sp::spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(_PTR_NAME_ =  reinterpret_cast<__TYPE__ *>(sp::spNext(&__it))) != 0x0;)

/**
 * insert elements to page .
 * example:
 * SP_ADD_NEW_ELEMENT(200,struct point_s, p, pg) {p->x = 0; }
 */
#define SP_PAGE_INSERT(__NUMBER__, __TYPE__, _PTR_NAME_, __PG_HEAD__)          \
__TYPE__ *_PTR_NAME_; size_t __count = __NUMBER__; \
for (sp::spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(_PTR_NAME_ =  reinterpret_cast<__TYPE__ *>(sp::spNextBlank(&__it))) != 0x0 && (__count>1);--__count)

#define SP_OBJ_REMOVE_IF(__TYPE__, _PTR_NAME_, __PG_HEAD__, __TAG__)          \
__TYPE__ *_PTR_NAME_ = 0x0; int __TAG__=0;\
for (sp::spIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(_PTR_NAME_ = reinterpret_cast<__TYPE__ *>( sp::spItRemoveIf(&__it,__TAG__))) != 0x0;)


//std::shared_ptr<spPagePool> makePagePool(size_t size_in_byte)
//{
//    return std::shared_ptr<spPagePool>(spPagePoolCreate(size_in_byte), [=](spPagePool *pg) { spPagePoolDestroy(&pg ); });
//}
//
//std::shared_ptr<spPage> makePage(std::shared_ptr<spPagePool> pool)
//{
//    return std::shared_ptr<spPage>(spPageCreate(pool.get()), [=](spPage *pg) { spPageClose(&pg, pool.get()); });
//}


}} //namespace simpla { namespace sp

#endif


#endif //SIMPLA_SMALLOBJPOOL_H_
