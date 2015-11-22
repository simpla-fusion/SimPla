/**
 * @file parallel_tbb.h
 *
 *  Created on: 2014-9-4
 *      Author: salmon
 */

#ifndef PARALLEL_TBB_H_
#define PARALLEL_TBB_H_

#include <tbb/tbb.h>

namespace simpla { namespace parallel
{
using tbb::parallel_for;
using tbb::parallel_do;
using tbb::parallel_reduce;

}}  // namespace simpla { namespace parallel


#endif /* PARALLEL_TBB_H_ */
