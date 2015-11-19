/**
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

//#elif _OPENMP
//#include "multi_thread_openmp.h"
#else

#include "multi_thread_std_thread.h"

#endif

namespace simpla
{

namespace tags
{
struct split;
}


namespace parallel
{
void init(int argc, char **argv);

void close();

std::string help_message();


template<typename TRange, typename Function>
void parallel_do(TRange const &range, Function const &fun)
{
    fun(TRange(range, tags::split()));
}

template<typename TRange, typename Function>
void parallel_for(TRange const &range, Function const &fun)
{
    parallel_do(range, [=](TRange const &o_range)
    {
        for (auto const &i:o_range)
        {
            fun(i);
        }
    });
}

template<typename TRange, typename Function, typename Reduction>
void parallel_reduce(TRange const &range, Function const &fun, Reduction const &reduction)
{
    parallel_do(range, [&](TRange const &o_range)
    {
        for (auto const &i:o_range)
        {
            fun(i);
        }
    });
}

} // namespace parallel
} // namespace simpla

#endif /* PARALLEL_H_ */
