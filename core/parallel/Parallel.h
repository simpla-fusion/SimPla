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
 *  	@addtogroup  MULTICORE Multi-thread/core and many-core support
 *  @}
 */
#ifndef NO_MPI

#include "MPIComm.h"

#endif

//#ifdef USE_TBB

#   include "ParallelTbb.h"
//#elif _OPENMP
//#include "multi_thread_openmp.h"
//#include "parallel_openmp.h"
//#else
////#include "multi_thread_std_thread.h"
//#include "ParallelDummy.h"
//
//#endif

#include "DistributedObject.h"

namespace simpla { namespace parallel
{
void init(int argc, char **argv);

void close();

std::string help_message();


template<typename ...Args>
void sync(Args &&...args)
{
    if (GLOBAL_COMM.num_of_process() > 1)
    {
        DistributedObject dist_obj;
        dist_obj.add(std::forward<Args>(args)...);
        dist_obj.sync();
        dist_obj.wait();
    }
};


}}// namespace simpla { namespace parallel

#endif /* PARALLEL_H_ */
