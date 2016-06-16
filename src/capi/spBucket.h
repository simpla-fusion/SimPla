/*
 * spBucket.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPBUCKET_H_
#define SPBUCKET_H_

#include "../sp_config.h"

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

typedef struct spPage_s
{
	struct spPage_s *next;
	bucket_entity_flag_t flag; // flag of element in the page, 'SP_NUMBER_OF_ENTITIES_IN_PAGE', 1 ->valid ,0 -> blank
	id_type tag; // tag of page group. In default, it storage 'bin id' for bucket sorting

	size_type entity_size_in_byte;
	byte_type *data;

} spPage;
struct spPagePool_s;
typedef struct spPagePool_s spPagePool;

typedef spPage bucket_type;

#endif /* SPBUCKET_H_ */
