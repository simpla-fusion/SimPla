/*
 * parallel.h
 *
 *  Created on: 2014年3月27日
 *      Author: salmon
 */

#ifndef PARALLEL_H_
#define PARALLEL_H_

#include "multi_thread.h"

#ifdef USE_MPI
#	include "message_comm.h"
#else
template<typename T>
void UpdateGhosts(T *, MPI_Comm comm = MPI_COMM_NULL)
{
}
#endif

/**
 *  \defgroup  Parallel Parallel
 *  @{
 *  	\defgroup  MPI MPI Communicaion
 *  	\defgroup  MULTICORE Multi-thread/core and many-core support
 *  @}
 */

#endif /* PARALLEL_H_ */
