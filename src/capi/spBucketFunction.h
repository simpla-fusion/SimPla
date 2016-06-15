//
// Created by salmon on 16-6-6.
//

#ifndef SIMPLA_SMALLOBJPOOL_H_
#define SIMPLA_SMALLOBJPOOL_H_

#include "spBucket.h"

/***************************************************************************/
/*  spPage Pool
 **/
void spPagePoolCreate(spPagePool **res, size_type size_in_byte);

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
spPage *spPageCreate(size_type num, spPagePool *pool);

/**
 * push page list back to pool
 * @return if success then *p=(p->next), return size of *p
 *                    else *p is not changed ,return 0
 */
size_type spPageDestroy(spPage **p, spPagePool *pool);

/**
 * insert an page to the beginning
 * @return if f!=0x0 then f->next=*p,   *p =f ,return f
 *                    else return *p
 */
spPage *spPagePushFront(spPage **p, spPage *f);
/**
 * remove first page from list
 * @return if success then *p=old *p ->next, return old *p
 *                    else *p is not changed ,return 0x0
 */
spPage *spPagePopFront(spPage **p);

/*-----------------------------------------------------------------------------*/
/* Element access */
/**
 *  access the first element
 */
spPage **spPageFront(spPage **p);

/**
 *  @return if success then  return the pointer to the last page
 *                      else  return 0x0
 */
spPage **spPageBack(spPage **p);
/*-----------------------------------------------------------------------------*/
/*  Capacity  */

/**
 * @return  the number of pages
 */
size_type spPageSize(spPage const *p);

/**
 *  checks whether the container is empty
 *  @return if empty return >0, else return 0
 */
int spPageIsEmpty(spPage const *p);

/**
 *  checks whether the container is full
 *  @return if every pages are full return >0, else return 0
 */
int spPageIsFull(spPage const *p);

/**
 * @return the number of elements that can be held in currently allocated storage
 */
size_type spPageCapacity(spPage const *p);
/**
 * @return  the number of entities in pages
 */
size_type spPageNumberOfEntities(spPage const *p);

/***************************************************************************/
/*  Entity
 **/

/**
 *  set page flag=0, do not change the capacity of pages
 */
void spEntityClear(spPage *p);

/**
 *  @return first entity after 'flag' , if flag=0x0 start from beginning
 *
 */
spEntity *spEntityNext(spPage **pg, bucket_entity_flag_t *flag);
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
spEntity *spEntityInsertWithHint(spPage **p, bucket_entity_flag_t *flag);

/**
 *  @return if success then return pointer to the first blank entity, and set flag to 1
 *                     else return 0x0
 */
spEntity *spEntityInsert(spPage *pg);

spEntity *spEntityAtomicInsert(spPage **pg, bucket_entity_flag_t *flag,
		spPagePool *pool);
/**
 * clear page, and fill N entities to page
 * @return number of remained entities.
 */
size_type spEntityFill(spPage *p, size_type num, const byte_type *src);

void spEntityRemove(spPage *p, bucket_entity_flag_t flag);

void spEntityCopyIf(spPage *src, spPage **dest, id_type tag, spPagePool *pool);

size_type spEntityCountIf(spPage *src, id_type tag);

#endif //SIMPLA_SMALLOBJPOOL_H_
