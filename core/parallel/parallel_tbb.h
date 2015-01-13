/*
 * parallel_tbb.h
 *
 *  Created on: 2014年9月4日
 *      Author: salmon
 */

#ifndef PARALLEL_TBB_H_
#define PARALLEL_TBB_H_

#include <tbb/tbb.h>
namespace simpla
{

using tbb::parallel_for;
using tbb::parallel_do;
using tbb::parallel_reduce;

}  // namespace simpla

#endif /* PARALLEL_TBB_H_ */
