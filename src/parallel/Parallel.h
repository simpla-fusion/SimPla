/**
 * @file parallel.h
 *
 *  created on: 2014-3-27
 *      Author: salmon
 */

#ifndef PARALLEL_H_
#define PARALLEL_H_

/**
 *  @addtogroup  parallel parallel
 *  @{
 *  	@addtogroup  MPI MPI Communicaion
 *  	@addtogroup  MULTICORE Multi-thread/src and many-src support
 *  @}
 */
#ifndef NO_MPI

#include "MPIComm.h"

#endif

#ifdef USE_TBB

#   include "ParallelTbb.h"

#elif _OPENMP

#   include "ParallelOpenMP.h"

#else

#   include "ParallelDummy.h"

#endif

#include "DistributedObject.h"

namespace simpla { namespace parallel
{
void init(int argc, char **argv);

void close();

std::string help_message();


}}// namespace simpla { namespace parallel

#endif /* PARALLEL_H_ */
