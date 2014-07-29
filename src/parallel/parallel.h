/*
 * parallel.h
 *
 *  created on: 2014-3-27
 *      Author: salmon
 */

#ifndef PARALLEL_H_
#define PARALLEL_H_

#include "multi_thread.h"

#ifdef USE_MPI
#	include "message_comm.h"
#endif

/**
 *  \defgroup  Parallel Parallel
 *  @{
 *  	\defgroup  MPI MPI Communicaion
 *  	\defgroup  MULTICORE Multi-thread/core and many-core support
 *  @}
 */

#endif /* PARALLEL_H_ */
