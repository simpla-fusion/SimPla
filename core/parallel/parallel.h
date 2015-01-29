/*
 * @file parallel.h
 *
 *  created on: 2014-3-27
 *      Author: salmon
 */

#ifndef PARALLEL_H_
#define PARALLEL_H_

/**
 *  @addtogroup  Parallel Parallel
 *  @{
 *  	@addtogroup  MPI MPI Communicaion
 *  	@addtogroup  MULTICORE Multi-thread/core and many-core support
 *  @}
 */
#ifndef NO_MPI
#include "mpi_comm.h"
#endif

#ifdef USE_TBB

#include "multi_thread_tbb.h"

#elif _OPENMP
#include "multi_thread_openmp.h"
#else
#include "multi_thread_std_thread.h"
#endif

namespace simpla
{

void init_parallel(int argc, char ** argv);
void close_parallel();

}  // namespace simpla

#endif /* PARALLEL_H_ */
