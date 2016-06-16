//
// Created by salmon on 16-6-6.
//

#ifndef SIMPLA_SMALLOBJPOOL_H_
#define SIMPLA_SMALLOBJPOOL_H_

#include "spBucket.h"

/***************************************************************************/
/*  spPage Pool
 **/
MC_HOST_DEVICE void spPagePoolCreate(spPagePool **res, size_type size_in_byte, size_type max_number_of_entity);

MC_HOST_DEVICE void spPagePoolDestroy(spPagePool **pool);

MC_HOST_DEVICE void spPagePoolReleaseEnpty(spPagePool *pool);

MC_HOST_DEVICE size_type spPagePoolEntitySizeInByte(spPagePool const *pool);

/***************************************************************************/
/*  spPage
 **/
/*-----------------------------------------------------------------------------*/
/*  Page create and modify */
/**
 *  pop 'num' pages from pool
 *  @return pointer of first page
 */
MC_HOST_DEVICE spPage *spPageCreate(size_type num, spPagePool *pool);

/**
 * push page list back to pool
 * @return if success then *p=(p->next), return size of *p
 *                    else *p is not changed ,return 0
 */
MC_HOST_DEVICE size_type spPageDestroy(spPage **p, spPagePool *pool);

/**
 * insert an page to the beginning
 * @return if f!=0x0 then f->next=*p,   *p =f ,return f
 *                    else return *p
 */
MC_HOST_DEVICE spPage *spPagePushFront(spPage **p, spPage *f);
/**
 * remove first page from list
 * @return if success then *p=old *p ->next, return old *p
 *                    else *p is not changed ,return 0x0
 */
MC_HOST_DEVICE spPage *spPagePopFront(spPage **p);

/*-----------------------------------------------------------------------------*/
/* Element access */
/**
 *  access the first element
 */
MC_HOST_DEVICE spPage **spPageFront(spPage **p);

/**
 *  @return if success then  return the pointer to the last page
 *                      else  return 0x0
 */
MC_HOST_DEVICE spPage **spPageBack(spPage **p);
/*-----------------------------------------------------------------------------*/
/*  Capacity  */

/**
 * @return  the number of pages
 */
MC_HOST_DEVICE size_type spPageSize(spPage const *p);

/**
 *  checks whether the container is empty
 *  @return if empty return >0, else return 0
 */
MC_HOST_DEVICE int spPageIsEmpty(spPage const *p);

/**
 *  checks whether the container is full
 *  @return if every pages are full return >0, else return 0
 */
MC_HOST_DEVICE int spPageIsFull(spPage const *p);

/**
 * @return the number of elements that can be held in currently allocated storage
 */
MC_HOST_DEVICE size_type spPageCapacity(spPage const *p);
/**
 * @return  the number of entities in pages
 */
MC_HOST_DEVICE size_type spPageNumberOfEntities(spPage const *p);

/***************************************************************************/
/*  Entity
 **/

/**
 *  set page flag=0, do not change the capacity of pages
 */
MC_HOST_DEVICE void spEntityClear(spPage *p);

/**
 *  @return first entity after 'flag' , if flag=0x0 start from beginning
 *
 */
MC_HOST_DEVICE spEntity *spEntityNext(spPage **pg, bucket_entity_flag_t *flag);
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
MC_HOST_DEVICE spEntity *spEntityInsertWithHint(spPage **p, bucket_entity_flag_t *flag);

/**
 *  @return if success then return pointer to the first blank entity, and set flag to 1
 *                     else return 0x0
 */
MC_HOST_DEVICE spEntity *spEntityInsert(spPage *pg);

MC_HOST_DEVICE spEntity *spEntityAtomicInsert(spPage **pg, bucket_entity_flag_t *flag, spPagePool *pool);
/**
 * clear page, and fill N entities to page
 * @return number of remained entities.
 */
MC_HOST_DEVICE size_type spEntityFill(spPage *p, size_type num, const byte_type *src);

MC_HOST_DEVICE void spEntityRemove(spPage *p, bucket_entity_flag_t flag);

MC_HOST_DEVICE void spEntityCopyIf(spPage *src, spPage **dest, id_type tag, spPagePool *pool);

MC_HOST_DEVICE size_type spEntityCountIf(spPage *src, id_type tag);

#endif //SIMPLA_SMALLOBJPOOL_H_
