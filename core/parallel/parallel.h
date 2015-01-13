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

#ifdef USE_TBB
#include <tbb/tbb.h>
namespace parallel = tbb;
#else
#include <atomic>
#include <mutex>
namespace parallel = std;
#endif

#include "mpi_comm.h"

namespace simpla
{

void init_parallel(int argc, char ** argv);
void close_parallel();

}  // namespace simpla

#endif /* PARALLEL_H_ */
