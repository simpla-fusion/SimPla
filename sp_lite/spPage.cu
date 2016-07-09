/*
 * spPage.cu
 *
 *  Created on: 2016年7月9日
 *      Author: salmon
 */
#include "spPage.h"
#include "spParallel.h"

MC_DEVICE spPage *spPageAtomicNext(spPage **pp)
{
	spPage *assumed;
	spPage *old = *pp;
	do
	{
		assumed = old;
		old = (spPage *) (atomicCAS((unsigned long long *) pp, (unsigned long long) assumed,
				(unsigned long long) (old->next)));
// Note: uses integer comparison to avoid hang in case of NaN (since NaN !=        NaN)
	} while (assumed != old);
	return (old);

}
