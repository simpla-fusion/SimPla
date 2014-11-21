/*
 * parallel.h
 *
 *  created on: 2014-3-27
 *      Author: salmon
 */

#ifndef PARALLEL_H_
#define PARALLEL_H_

//#include "multi_thread.h"

/**
 *  \defgroup  Parallel Parallel
 *  @{
 *  	\defgroup  MPI MPI Communicaion
 *  	\defgroup  MULTICORE Multi-thread/core and many-core support
 *  @}
 */
#ifdef _OPENMP
#	include "parallel_openmp.h"
#elif defined(USE_TBB) &&  USE_TBB==on
#	include "parallel_tbb.h"
#else
#	include "parallel_dummy.h"
#endif

namespace simpla
{
void init_parallel(int argc, char ** argv);
void close_parallel();

}  // namespace simpla

#endif /* PARALLEL_H_ */
