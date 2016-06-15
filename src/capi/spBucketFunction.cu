//
// Created by salmon on 16-6-14.
//
#include "../sp_config.h"

#include "sp_def_cuda.h"
#include "spBucketFunction.h"

typedef struct spPageGroup_s
{
	struct spPageGroup_s *next;
	size_type number_of_pages;
	spPage *m_pages;
	byte_type *m_data;
} spPageGroup;

typedef struct spPagePool_s
{
	size_type entity_size_in_byte;
	spPageGroup *m_page_group_head;
	spPage *m_free_page;
	pthread_mutex_t m_pool_mutex_;

} spPagePool;
#define DEFAULT_NUMBER_OF_PAGES_IN_GROUP 64
/*******************************************************************************************/

__device__ size_type spPagePoolEntitySizeInByte(spPagePool const *pool)
{
	return pool->entity_size_in_byte;
}

__device__ spPageGroup *spPageGroupCreate(size_type entity_size_in_byte,
		size_type num_of_pages)
{
	spPageGroup *res = 0x0;

	res = (spPageGroup *) (malloc(sizeof(spPageGroup)));
	res->m_pages = (spPage*) malloc(sizeof(spPage) * num_of_pages);
	res->m_data = (byte_type*) malloc(
			entity_size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE * num_of_pages);

	res->next = 0x0;
	res->number_of_pages = num_of_pages;

	for (int i = 0; i < num_of_pages; ++i)
	{
		res->m_pages[i].next = &(res->m_pages[i + 1]);
		res->m_pages[i].flag = 0x0;
		res->m_pages[i].tag = 0x0;
		res->m_pages[i].entity_size_in_byte = entity_size_in_byte;
		res->m_pages[i].data = res->m_data
				+ i * (entity_size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE);
	}
	res->m_pages[num_of_pages - 1].next = 0x0;
	return res;
}

/**
 * @return next page group
 */
__device__
void spPageGroupDestroy(spPageGroup **pg)
{
	if (pg != 0 && *pg != 0)
	{
		spPageGroup *t = (*pg);
		(*pg) = (*pg)->next;

		free(t->m_data);
		free(t->m_pages);
		free(t);

	}
}

/**
 *  @return first free page
 *    pg = first page group
 */
__device__ size_type spPageGroupSize(spPageGroup const *pg)
{

	size_type count = 0;

	for (int i = 0; i < pg->number_of_pages; ++i)
	{
		count += spPageNumberOfEntities(&(pg->m_pages[i]));
	}
	return count;
}

__device__
void spPagePoolCreate(spPagePool **res, size_type size_in_byte)
{
	*res = 0x0;

	*res = (spPagePool *) (malloc(sizeof(spPagePool)));
	(*res)->entity_size_in_byte = size_in_byte;
	(*res)->m_page_group_head = 0x0;
	(*res)->m_free_page = 0x0;

}

__device__
void spPagePoolDestroy(spPagePool **pool)
{

	while ((*pool)->m_page_group_head != 0x0)
	{
		spPageGroup *pg = (*pool)->m_page_group_head;
		(*pool)->m_page_group_head = (*pool)->m_page_group_head->next;
		spPageGroupDestroy(&pg);
	}
	free(*pool);
	(*pool) = 0x0;

}

__device__
void spPagePoolReleaseEnpty(spPagePool *pool)
{
	spPageGroup *head = pool->m_page_group_head;
	while (head != 0x0)
	{
		if (spPageGroupSize(head) == 0)
		{
			spPageGroupDestroy(&head);
		}
		else
		{
			head = head->next;
		}
	}
}

/****************************************************************************
 *  Page create and modify
 */

__device__ spPage *spPageCreate(size_type num, spPagePool *pool)
{
//	pthread_mutex_lock(&(pool->m_pool_mutex_));
	spPage *head = 0x0;
	spPage **tail = &(pool->m_free_page);
	while (num > 0)
	{
		if ((*tail) == 0x0)
		{
			spPageGroup *pg = spPageGroupCreate(pool->entity_size_in_byte,
			DEFAULT_NUMBER_OF_PAGES_IN_GROUP);
			pg->next = pool->m_page_group_head;
			pool->m_page_group_head = pg;
			(*tail) = &((pool->m_page_group_head->m_pages)[0]);
		}
		if (head == 0x0)
		{
			head = (*tail);
		}

		while (num > 0 && (*tail) != 0x0)
		{
			tail = &((*tail)->next);
			--num;
		}

	}
	pool->m_free_page = (*tail)->next;
	(*tail)->next = 0x0;
//	pthread_mutex_unlock(&(pool->m_pool_mutex_));
	return head;
}

__device__ size_t spPageDestroy(spPage **p, spPagePool *pool)
{
//	pthread_mutex_lock(&(pool->m_pool_mutex_));

	size_type res = spPageSize(*p);

	spPagePushFront(&(pool->m_free_page), spPagePopFront(p));

//	pthread_mutex_unlock(&(pool->m_pool_mutex_));

	return res;
}

__device__ spPage * spPagePushFront(spPage **p, spPage *f)
{
	if (f != 0x0)
	{
		*spPageBack(&f) = *p;
		*p = f;
	}

	return *p;

}

__device__ spPage *
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

/****************************************************************************
 * Element access
 */

__device__ spPage **
spPageFront(spPage **p)
{
	return p;
}

__device__ spPage **
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
__device__ size_type spPageSize(spPage const *p)
{
	size_type res = 0;
	while (p != 0x0)
	{
		++res;
		p = p->next;
	}
	return res;
}

__device__ int spPageIsEmpty(spPage const *p)
{
	int count = 0;
	while (p != 0x0)
	{
		count += (p->flag != 0x0) ? 1 : 0;
		p = p->next;
	}

	return (count > 0) ? 0 : 1;
}

__device__ int spPageIsFull(spPage const *p)
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

__device__ size_type bit_count64(uint64_t x)
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

__device__ size_type spPageNumberOfEntities(spPage const *p)
{
	size_type res = 0;
	while (p != 0x0)
	{
		res += bit_count64(p->flag);
		p = p->next;
	}
	return res;
}

__device__ size_type spPageCapacity(spPage const *p)
{
	return spPageSize(p) * SP_NUMBER_OF_ENTITIES_IN_PAGE;
}

/***************************************************************************/
/*  Entity
 **/

__device__ void spEntityClear(spPage *p)
{
	while (p != 0x0)
	{
		p->flag = 0x0;
		p = p->next;
	}
}
;

__device__ size_type spEntityFill(spPage *p, size_type num,
		const byte_type *src)
{
	while (num > 0 && p != 0x0)
	{
		size_type n =
				(num < SP_NUMBER_OF_ENTITIES_IN_PAGE) ?
						num : SP_NUMBER_OF_ENTITIES_IN_PAGE;

		memcpy(p->data, src, p->entity_size_in_byte * n);

		src += p->entity_size_in_byte * n;

		num -= n;

		p->flag = (bucket_entity_flag_t) (0 - 1);
		p = p->next;
	}
	return num;

}

__device__ spEntity *
spEntityInsert(spPage *pg)
{
	spPage *t = pg;
	bucket_entity_flag_t flag = 0x0;
	return spEntityInsertWithHint(&t, &flag);
}

__device__ spEntity *
spEntityInsertWithHint(spPage **pg, bucket_entity_flag_t *flag)
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

			res += (*pg)->entity_size_in_byte;
			*flag <<= 1;

		}

		*flag = 0x1;
		pg = &(*pg)->next;

	}
	RETURN: return (spEntity *) res;
}

__device__ spEntity *
spEntityNext(spPage **pg, bucket_entity_flag_t *flag)
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

			res += (*pg)->entity_size_in_byte;
			*flag <<= 1;

		}

		*flag = 0x1;
		pg = &(*pg)->next;

	}
	RETURN: return (spEntity *) res;
}

__device__ void spEntityRemove(spPage *p, bucket_entity_flag_t flag)
{
	p->flag &= (~flag);
}

#ifndef DEFAULT_COPY
#   define DEFAULT_COPY(_SRC_, _DEST_)  memcpy(_DEST_,_SRC_,entity_size_in_byte)
#endif

__device__ size_type spEntityCountIf(spPage *src, id_type tag)
{

	size_type count = 0;

	spPage *pg = src;

	bucket_entity_flag_t read_flag = 0x0;

	for (spEntity *p; (p = spEntityNext(&pg, &read_flag)) != 0x0;)
	{
		if ((p->_tag & 0x3F) == tag)
		{
			++count;
		}
	}
	return count;
}

__device__ void spEntityCopyIf(spPage *src, spPage **dest, id_type tag,
		spPagePool *pool)
{

	spPage *pg = src;

	size_type entity_size_in_byte = spPagePoolEntitySizeInByte(pool);

	bucket_entity_flag_t read_flag = 0x0;

	spPage *write_buffer = 0x0;

	bucket_entity_flag_t write_flag = 0x0;

	for (spEntity *p0, *p1 = 0x0; (p0 = spEntityNext(&pg, &read_flag)) != 0x0;)
	{
		if ((p0->_tag & 0x3F) == tag)
		{
			if (write_flag == 0x0 || write_buffer == 0x0)
			{
				/** atomic */
				spPagePushFront(dest, write_buffer);
				/** atomic */
				write_buffer = spPageCreate(1, pool);
				write_flag = 0x0;
				p1 = (spEntity *) write_buffer->data;
			}

			DEFAULT_COPY(p0, p1);
			p1->_tag &= ~(0x3F); // clear tag
			write_buffer->flag |= write_flag;
			write_flag <<= 1;
			p1 = (spEntity *) (((byte_type *) p1) + entity_size_in_byte);

		}
	}
}

__device__ int spBucketEnternalSort(spPage **src, spPage **dest)
{
	return 0;
}
