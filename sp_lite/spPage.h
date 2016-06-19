/*
 * spBucket.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPBUCKET_H_
#define SPBUCKET_H_

#include "sp_def.h"

// digits of bucket_page_status_flag_t
#define SP_NUMBER_OF_ENTITIES_IN_PAGE 64

typedef uint64_t bucket_entity_flag_t;

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
#define SP_BUCKET_ENTITY_HEAD bucket_entity_flag_t  tag;

typedef struct spEntity_s
{
	SP_BUCKET_ENTITY_HEAD
	byte_type data[];
} spEntity;
#define SP_PAGE_HEAD struct spPage_s *next;	bucket_entity_flag_t flag;
typedef struct spPage_s
{
	SP_PAGE_HEAD
	byte_type *data;
} spPage;

/***************************************************************************/
/**
 * insert an page to the beginning
 * @return if f!=0x0 then f->next=*p,   *p =f ,return f
 *                    else return *p
 */
MC_HOST_DEVICE extern inline spPage *spPagePushFront(spPage **p, spPage *f);
/**
 * remove first page from list
 * @return if success then *p=old *p ->next, return old *p
 *                    else *p is not changed ,return 0x0
 */
MC_HOST_DEVICE extern inline spPage *spPagePopFront(spPage **p);

/*-----------------------------------------------------------------------------*/
/* Element access */
/**
 *  access the first element
 */
MC_HOST_DEVICE extern inline spPage **spPageFront(spPage **p);

/**
 *  @return if success then  return the pointer to the last page
 *                      else  return 0x0
 */
MC_HOST_DEVICE extern inline spPage **spPageBack(spPage **p);
/*-----------------------------------------------------------------------------*/
/*  Capacity  */

/**
 * @return  the number of pages
 */
MC_HOST_DEVICE extern inline size_type spPageSize(spPage const *p);

/**
 *  checks whether the container is empty
 *  @return if empty return >0, else return 0
 */
MC_HOST_DEVICE extern inline int spPageIsEmpty(spPage const *p);

/**
 *  checks whether the container is full
 *  @return if every pages are full return >0, else return 0
 */
MC_HOST_DEVICE extern inline int spPageIsFull(spPage const *p);

/**
 * @return the number of elements that can be held in currently allocated storage
 */
MC_HOST_DEVICE extern inline size_type spPageCapacity(spPage const *p);
/**
 * @return  the number of entities in pages
 */
MC_HOST_DEVICE extern inline size_type spPageNumberOfEntities(spPage const *p);

/***************************************************************************/
/*  Entity
 **/

/**
 *  set page flag=0, do not change the capacity of pages
 */
MC_HOST_DEVICE extern inline void spEntityClear(spPage *p);

MC_HOST_DEVICE extern inline spEntity *spEntityInsertWithHint(spPage **p, bucket_entity_flag_t *flag,
		size_type entity_size_in_byte);

MC_HOST_DEVICE extern inline spEntity *spEntityAtomicInsert(spPage **pg, bucket_entity_flag_t *flag,
		size_type entity_size_in_byte);
/**
 * clear page, and fill N entities to page
 * @return number of remained entities.
 */
MC_HOST_DEVICE extern inline size_type spEntityFill(spPage *p, size_type num, const byte_type *src,
		size_type entity_size_in_byte);

MC_HOST_DEVICE extern inline void spEntityRemove(spPage *p, bucket_entity_flag_t flag);

MC_HOST_DEVICE extern inline void spEntityCopyIf(spPage *src, spPage **dest, id_type tag);

MC_HOST_DEVICE extern inline size_type spEntityCountIf(spPage *src, id_type tag);

/****************************************************************************
 *  Page create and modify
 */

MC_HOST_DEVICE extern inline spPage *
spPagePushFront(spPage **p, spPage *f)
{
	if (f != 0x0)
	{
		*spPageBack(&f) = *p;
		*p = f;
	}

	return *p;

}

MC_HOST_DEVICE extern inline spPage *
spPagePopFront(spPage **p)
{
	spPage *res = 0x0;
	if (p != 0x0 && *p != 0x0)
	{
		res = *p;
		*p = (*p)->next;

		res->next = 0x0;
	}
	return res;
}
MC_HOST_DEVICE extern inline spPage *
spPagePopFrontN(spPage **p, size_type num)
{
	spPage *res = 0x0;
	if (p != 0x0 && *p != 0x0)
	{
		res = *p;
		*p = (*p)->next;

		res->next = 0x0;
	}
	return res;
}

/****************************************************************************
 * Element access
 */

MC_HOST_DEVICE extern inline spPage **
spPageFront(spPage **p)
{
	return p;
}

MC_HOST_DEVICE extern inline spPage **
spPageBack(spPage **p)
{
	while (p != 0x0 && *p != 0x0 && (*p)->next != 0x0)
	{
		p = &((*p)->next);
	}
	return p;
}

/****************************************************************************
 * Capacity
 */
MC_HOST_DEVICE extern inline size_type spPageSize(spPage const *p)
{
	size_type res = 0;
	while (p != 0x0)
	{
		++res;
		p = p->next;
	}
	return res;
}

MC_HOST_DEVICE extern inline int spPageIsEmpty(spPage const *p)
{
	int count = 0;
	while (p != 0x0)
	{
		count += (p->flag != 0x0) ? 1 : 0;
		p = p->next;
	}

	return (count > 0) ? 0 : 1;
}

MC_HOST_DEVICE extern inline int spPageIsFull(spPage const *p)
{
	if (p == 0x0)
	{
		return 0;
	}
	else
	{
		int count = 0;
		while (p != 0x0)
		{
			count += ((p->flag + 1) != 0x0) ? 1 : 0;
			p = p->next;
		}
		return count;
	}
}

MC_HOST_DEVICE extern inline size_type bit_count64(uint64_t x)
{
#define m1   0x5555555555555555
#define m2   0x3333333333333333
#define m4   0x0f0f0f0f0f0f0f0f
#define m8   0x00ff00ff00ff00ff
#define m16  0x0000ffff0000ffff
#define m32  0x00000000ffffffff

	x = (x & m1) + ((x >> 1) & m1); //put count of each  2 bits into those  2 bits
	x = (x & m2) + ((x >> 2) & m2); //put count of each  4 bits into those  4 bits
	x = (x & m4) + ((x >> 4) & m4); //put count of each  8 bits into those  8 bits
	x = (x & m8) + ((x >> 8) & m8); //put count of each 16 bits into those 16 bits
	x = (x & m16) + ((x >> 16) & m16); //put count of each 32 bits into those 32 bits
	x = (x & m32) + ((x >> 32) & m32); //put count of each 64 bits into those 64 bits
	return (size_type) x;

}

MC_HOST_DEVICE extern inline size_type spPageNumberOfEntities(spPage const *p)
{
	size_type res = 0;
	while (p != 0x0)
	{
		res += bit_count64(p->flag);
		p = p->next;
	}
	return res;
}

MC_HOST_DEVICE extern inline size_type spPageCapacity(spPage const *p)
{
	return spPageSize(p) * SP_NUMBER_OF_ENTITIES_IN_PAGE;
}

/***************************************************************************/
/*  Entity
 **/

MC_HOST_DEVICE extern inline void spEntityClear(spPage *p)
{
	while (p != 0x0)
	{
		p->flag = 0x0;
		p = p->next;
	}
}
/**
 *  @return if success then return pointer to the first blank entity, and set flag to 1
 *                     else return 0x0
 */
MC_HOST_DEVICE extern inline spEntity *
spEntityInsert(spPage *pg, size_type entity_size_in_byte)
{
	spPage *t = pg;
	bucket_entity_flag_t flag = 0x0;
	return spEntityInsertWithHint(&t, &flag, entity_size_in_byte);
}
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
MC_HOST_DEVICE extern inline spEntity *
spEntityInsertWithHint(spPage **pg, bucket_entity_flag_t *flag, size_type entity_size_in_byte)
{
	byte_type *res = 0x0;
	if (*flag == 0x0)
	{
		*flag = 0x1;
	}

	while ((*pg) != 0)
	{
		res = (*pg)->data;

		while (((*pg)->flag + 1 != 0x0) && *flag != 0x0)
		{
			if (((*pg)->flag & *flag) == 0x0)
			{
				(*pg)->flag |= *flag;
				goto RETURN;
			}

			res += entity_size_in_byte;
			*flag <<= 1;

		}

		*flag = 0x1;
		pg = &(*pg)->next;

	}
	RETURN: return (spEntity *) res;
}
/**
 *  @return first entity after 'flag' , if flag=0x0 start from beginning
 *
 */
MC_HOST_DEVICE extern inline spEntity *
spEntityNext(spPage **pg, bucket_entity_flag_t *flag, size_type entity_size_in_byte)
{

	byte_type *res = 0x0;
	if (*flag == 0x0)
	{
		*flag = 0x1;
	}

	while ((*pg) != 0)
	{
		res = (*pg)->data;

		while (((*pg)->flag != 0x0) && *flag != 0x0)
		{
			if (((*pg)->flag & *flag) != 0x0)
			{
				goto RETURN;
			}

			res += entity_size_in_byte;
			*flag <<= 1;

		}

		*flag = 0x1;
		pg = &(*pg)->next;

	}
	RETURN: return (spEntity *) res;
}

MC_HOST_DEVICE extern inline void spEntityRemove(spPage *p, bucket_entity_flag_t flag)
{
	p->flag &= (~flag);
}

#ifndef DEFAULT_COPY
#   define DEFAULT_COPY(_SRC_, _DEST_)  memcpy(_DEST_,_SRC_,entity_size_in_byte)
#endif

MC_HOST_DEVICE extern inline size_type spEntityCountIf(spPage *src, id_type tag, size_type entity_size_in_byte)
{

	size_type count = 0;

	spPage *pg = src;

	bucket_entity_flag_t read_flag = 0x0;

	for (spEntity *p; (p = spEntityNext(&pg, &read_flag, entity_size_in_byte)) != 0x0;)
	{
		if ((p->tag & 0x3F) == tag)
		{
			++count;
		}
	}
	return count;
}

MC_HOST_DEVICE extern inline void spEntityCopyIf(spPage *src, spPage **dest, id_type tag, size_type entity_size_in_byte)
{

	spPage *pg = src;

	bucket_entity_flag_t read_flag = 0x0;

	spPage *write_buffer = 0x0;

	bucket_entity_flag_t write_flag = 0x0;

	for (spEntity *p0, *p1 = 0x0; (p0 = spEntityNext(&pg, &read_flag, entity_size_in_byte)) != 0x0;)
	{
		if ((p0->tag & 0x3F) == tag)
		{
			if (write_flag == 0x0 || write_buffer == 0x0)
			{
				break;
			}

			DEFAULT_COPY(p0, p1);
			p1->tag &= ~(0x3F); // clear tag
			write_buffer->flag |= write_flag;
			write_flag <<= 1;
			p1 = (spEntity *) (((byte_type *) p1) + entity_size_in_byte);

		}
	}
}

MC_HOST_DEVICE extern inline int spBucketEnternalSort(spPage **src, spPage **dest)
{
	return 0;
}
#endif /* SPBUCKET_H_ */
