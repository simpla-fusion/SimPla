//
// Created by salmon on 16-6-6.
//

#ifndef SIMPLA_SMALLOBJPOOL_H_
#define SIMPLA_SMALLOBJPOOL_H_

#include "../sp_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>


enum { SP_SUCCESS, SP_BUFFER_EMPTY };


#define BUCKET_ELEMENT_HEAD uint64_t _tag;
/**
 * bucket_elements_head
 *  uint64_t shift;
 *   0b0000000000000000000000zzyyxx
 *   first 38 bits is not defined for 'bucket_elements_head',
 *       (for 'particle', they are constant index number of particle),
 *   last  6 bits 'zzyyxx' are relative cell offsets
 *         000000 means particle is the correct cell
 *         if xx = 00 means particle is the correct cell
 *                 01 -> (+1) right neighbour cell
 *                 11 -> (-1)left neighbour cell
 *                 10  not neighbour, if xx=10 , r[0]>2 or r[0]<-1
 *
 *        |   001010   |    001000     |   001001      |
 * -------+------------+---------------+---------------+---------------
 *        |            |               |               |
 *        |            |               |               |
 * 000011 |   000010   |    000000     |   000001      |   000011
 *        |            |               |               |
 *        |            |               |               |
 * -------+------------+---------------+---------------+---------------
 *        |   000110   |    000100     |   000101      |
 *
 *  r    : local coordinate r\in [0,1]
 *
 *
 *               11             00              01
 * ---------+------------+-------@--------+-------------+---------------
 */
struct spBucketElementHead
{
    BUCKET_ELEMENT_HEAD
    byte_type *data;
};
// digits of status_flag_type
#define SP_NUMBER_OF_ELEMENT_IN_PAGE 64
typedef uint64_t status_flag_type;

struct spPage
{
    struct spPage *next;
    id_type tag;   // tag of page group. In default, it storage 'bin id' for bucket sorting
    status_flag_type flag; // flag of element in the page, 'SP_NUMBER_OF_ELEMENT_IN_PAGE', 1 ->valid ,0 -> blank

    size_type ele_size_in_byte;
    byte_type *data;

};
struct spPagePool;

/**
 * @brief Resort elements in buckets. Elements in same bucket are unordered.
 *        assume the distance from element to its bucket is less or equal one .
 * @param buckets    array of buckets
 * @param ndims      number of dimensions of bucket array
 * @param dims       dimensions of bucket array
 * @param pool       pool of pages
 * @return
 */
void spBucketResort(struct spPage **buckets, int ndims, size_type const *dims, struct spPagePool *pool);

/****************************************************************************/



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
size_t spPageDestroyN(struct spPagePool *pool, struct spPage **p, size_t num);

void spPageDestroy(struct spPagePool *pool, struct spPage **p);



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
void spSetPageFlag(struct spPage *p, size_t tag);

//
void spPushFront(struct spPage **from, struct spPage **to);

void spPopFront(struct spPage **from, struct spPage **to);
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
size_t spInsert(struct spPage *p, size_t N, size_t size_in_byte, const byte_type *src);

byte_type *spInsertOne(struct spPage **p);

size_t spFill(struct spPage *p, size_t N, size_t size_in_byte, const byte_type *src);
/****************************************************************************
 * Operations
 */
/**
 * merges two sorted lists
 */


/****************************************************************************
 * Iterators
 */
struct spOutputIterator
{
    status_flag_type flag;
    void *const p;
    struct spPage **const page;
    size_t ele_size_in_byte;
};

/**
 * 1. move 'it' to 'next obj'
 *
 * @return  if 'next obj' is available then return pointer to 'next obj'
 *           else return 0x0
 */
void *spNext(struct spOutputIterator *it);

struct spInputIterator
{
    status_flag_type tag;
    void *p;
    struct spPage **page;
    struct spPagePool *pool;
};

/**
 * 1. insert new obj to page
 * @return if page is full return 0x0, else return pointer to new obj
 */
void *spInputIteratorNext(struct spInputIterator *it);


size_t spNextBlank2(struct spPage **pg, size_t *tag, byte_type **p, struct spPagePool *pool);


/**
 * 1. if flag > 0 then remove 'current obj', else do nothing
 * 2. move 'it' to next obj
 * @return if 'next obj' is available then return pointer to 'next obj',
 *          else return 0x0
 */
void *spItRemoveIf(struct spOutputIterator *it, int flag);


struct spPagePool;

struct spPagePool *spPagePoolCreate(size_t size_in_byte);

void spPagePoolDestroy(struct spPagePool **pool);

void spPagePoolRelease(struct spPagePool *pool);


size_t spSizeInByte(struct spPagePool const *pool);

/**
 *   traverses all element
 *   example:
 *       SP_FOREACH_ELEMENT(struct point_s, p, pg){ p->x = 0;}
 */
#define SP_PAGE_FOREACH(__TYPE__, _PTR_NAME_, __PG_HEAD__)          \
__TYPE__ *_PTR_NAME_ = 0x0; \
for (struct spOutputIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(_PTR_NAME_ =(__TYPE__ *)  spNext(&__it)) != 0x0;)

/**
 * insert elements to page .
 * example:
 * SP_ADD_NEW_ELEMENT(200,struct point_s, p, pg, p_pool) {p->x = 0; }
 */
#define SP_PAGE_INSERT_PTR(__NUMBER__, __TYPE__, _PTR_NAME_, __PG_HEAD__, __POOL__)          \
__TYPE__ *_PTR_NAME_; size_t __count = __NUMBER__; \
for (struct spInputIterator __it = {0x0, 0x0, __PG_HEAD__, __POOL__}; \
(_PTR_NAME_ =  spNextBlank(&__it )) != 0x0 && (__count>1);--__count)

#define SP_OBJ_REMOVE_IF(__TYPE__, _PTR_NAME_, __PG_HEAD__, __TAG__)          \
__TYPE__ *_PTR_NAME_ = 0x0; int __TAG__=0;\
for (struct spOutputIterator __it = {0x0, 0x0, __PG_HEAD__, sizeof(__TYPE__)}; \
(_PTR_NAME_ =  spItRemoveIf(&__it,__TAG__)) != 0x0;)

#ifdef __cplusplus
}// extern "C"
#endif


#endif //SIMPLA_SMALLOBJPOOL_H_
