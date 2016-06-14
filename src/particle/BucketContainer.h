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


#define SP_ENTITY_HEAD uint64_t _tag;
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
typedef struct spEntity_s
{
    SP_ENTITY_HEAD
    byte_type *data;
} spEntity;
// digits of bucket_page_status_flag_t
#define SP_NUMBER_OF_ELEMENT_IN_PAGE 64
typedef uint64_t bucket_page_status_flag_t;

typedef struct spPage_s
{
    struct spPage_s *next;
    bucket_page_status_flag_t flag; // flag of element in the page, 'SP_NUMBER_OF_ELEMENT_IN_PAGE', 1 ->valid ,0 -> blank
    id_type tag;   // tag of page group. In default, it storage 'bin id' for bucket sorting

    size_type entity_size_in_byte;
    byte_type *data;

} spPage;
struct spPagePool_s;
typedef struct spPagePool_s spPagePool;


/***************************************************************************/
/*  spPage Pool
 **/


spPagePool *spPagePoolCreate(size_type entity_size_in_byte);

void spPagePoolDestroy(spPagePool **pool);

void spPagePoolReleaseEnpty(spPagePool *pool);

size_type spPagePoolEntitySizeInByte(spPagePool const *pool);

/***************************************************************************/
/*  spPage
 **/
/*-----------------------------------------------------------------------------*/
/*  Page create and modify */
/**
 *  pop 'num' pages from pool
 *  @return pointer of first page
 */
MC_DEVICE MC_HOST spPage *spPageCreate(size_type num, spPagePool *pool);

/**
 * push page list back to pool
 * @return if success then *p=(p->next), return size of *p
 *                    else *p is not changed ,return 0
 */
MC_DEVICE MC_HOST size_type spPageDestroy(spPage **p, spPagePool *pool);


/**
 * insert an page to the beginning
 * @return if success then *p=old *p ->next, return old *p
 *                    else *p is not changed ,return 0x0
 */
MC_DEVICE MC_HOST void spPagePushFront(spPage **p, spPage *f);
/**
 * remove first page from list
 * @return if success then *p=old *p ->next, return old *p
 *                    else *p is not changed ,return 0x0
 */
MC_DEVICE MC_HOST spPage *spPagePopFront(spPage **p);

/**
 *  transfers one page from one list to an other
 *   move the front page from 'other' to 'self', and push front
 *  @return if success then *self = (old *other), (*self)->next=old (*self)
 *                          *other =(old *other) ->next
 *                           return 1
 *                     else  *self,*other is not changed
 *                           return 0
 */
MC_DEVICE MC_HOST size_type spPageSplice(spPage **self, spPage **other);


/**
 *  transfers one list to another
 *   move the front page from 'other' to 'self', and push front
 */
MC_DEVICE MC_HOST size_type spPageSpliceAll(spPage **self, spPage **other);

/*-----------------------------------------------------------------------------*/
/* Element access */
/**
 *  access the first element
 */
MC_DEVICE MC_HOST spPage **spPageFront(spPage **p);

/**
 *  @return if success then  return the pointer to the last page
 *                      else  return 0x0
 */
MC_DEVICE MC_HOST spPage **spPageBack(spPage **p);
/*-----------------------------------------------------------------------------*/
/*  Capacity  */

/**
 * @return  the number of pages
 */
MC_DEVICE MC_HOST size_type spPageSize(spPage const *p);


/**
 *  checks whether the container is empty
 *  @return if empty return >0, else return 0
 */
MC_DEVICE MC_HOST int spPageIsEmpty(spPage const *p);


/**
 *  checks whether the container is full
 *  @return if every pages are full return >0, else return 0
 */
MC_DEVICE MC_HOST int spPageIsFull(spPage const *p);



/**
 * @return the number of elements that can be held in currently allocated storage
 */
MC_DEVICE MC_HOST size_type spPageCapacity(spPage const *p);
/**
 * @return  the number of entities in pages
 */
MC_DEVICE MC_HOST size_type spPageNumberOfEntities(spPage const *p);

/***************************************************************************/
/*  Entity
 **/

/**
 *  set page flag=0, do not change the capacity of pages
 */
MC_DEVICE MC_HOST void spEntityClear(spPage *p);

/**
 *  @return first entity after 'flag' , if flag=0x0 start from beginning
 *
 */
MC_DEVICE MC_HOST spEntity *spEntityNext(spPage **pg, bucket_page_status_flag_t *flag);
/**
 *  find first blank entity
 *  @param  flag  search from the 'flag'
 *           (default: flag=0x0, start from beginning),
 *
 *  @return if success then *p point to the page of result, flag point the position of result
 *                           and set flag to 1
 *                     else return 0x0, *p ,flag is undefined
 *
 */
MC_DEVICE MC_HOST spEntity *spEntityInsertWithHint(spPage **p, bucket_page_status_flag_t *flag);

/**
 *  @return if success then return pointer to the first blank entity, and set flag to 1
 *                     else return 0x0
 */
MC_DEVICE MC_HOST spEntity *spEntityInsert(spPage *pg);
/**
 * clear page, and fill N entities to page
 * @return number of remained entities.
 */
MC_DEVICE MC_HOST size_type spEntityFill(spPage *p, size_type num, const byte_type *src);

MC_DEVICE MC_HOST void spEntityRemove(spPage *p, bucket_page_status_flag_t flag);


/***************************************************************************/
/*  Algorithm
 **/

void spBucketResort(spPage **buckets, int ndims, size_type const *dims, spPagePool *pool);


#ifdef __cplusplus
}// extern "C"
#endif


#endif //SIMPLA_SMALLOBJPOOL_H_
