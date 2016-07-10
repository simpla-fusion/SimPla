/*
 * spPage.cu
 *
 *  Created on: 2016年7月10日
 *      Author: salmon
 */
#include <assert.h>
#include "spParallel.h"
#include "spPage.h"

MC_DEVICE spPage* spPageAtomicPop(spPage **pg)
{
	spPage_s * old = (*pg);
	spPage_s ** address_as_ull = pg;
	spPage_s *assumed, *next;
	assert(sizeof(unsigned long long int) == sizeof(struct spPapge_s *));
	do
	{
		if (old == NULL)
		{
			break;
		}

		assumed = old;

		next = (spPage*) (old->next);

		old = (spPage*) atomicCAS((unsigned long long int*) address_as_ull, (unsigned long long int) (assumed),
				(unsigned long long int) (next));
	} while (assumed != old);

	return old;

}

MC_DEVICE spPage* spPageAtomicPush(spPage **pg, spPage* v)
{
	spPage_s * old = (*pg);
//	spPage_s ** address_as_ull = pg;
//	spPage_s *assumed, *next;
//	assert(sizeof(unsigned long long int) == sizeof(struct spPapge_s *));
//	do
//	{
//		if (old == NULL)
//		{
//			break;
//		}
//
//		assumed = old;
//
//		next = (spPage*) (old->next);
//
//		old = (spPage*) atomicCAS((unsigned long long int*) address_as_ull, (unsigned long long int) (assumed),
//				(unsigned long long int) (next));
//	} while (assumed != old);

	return old;

}
