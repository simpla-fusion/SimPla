/**
 * @file multi_thread_tbb.h
 *
 * @date    2014-9-5  AM6:48:54
 * @author salmon
 */

#ifndef MULTI_THREAD_TBB_H_
#define MULTI_THREAD_TBB_H_
#include <tbb/tbb.h>
namespace simpla
{
using tbb::parallel_for;
using tbb::parallel_do;
using tbb::parallel_reduce;

}  // namespace simpla

#endif /* MULTI_THREAD_TBB_H_ */
