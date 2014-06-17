/*
 * multi_thread.h
 *
 *  Created on: 2014年5月12日
 *      Author: salmon
 */

#ifndef MULTI_THREAD_H_
#define MULTI_THREAD_H_
#include <vector>

#ifdef _OPENMP
#	include <omp.h>
#else
#	include <thread>
#endif
namespace simpla
{

inline void ParallelDo(std::function<void(int, int)> fun)
{

#ifdef _OPENMP
	int num_threads = omp_get_num_procs();

#pragma omp parallel for
	for ( int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		fun(num_threads, thread_id);
	}
#else
	const unsigned int num_threads = GLOBAL_COMM.GetNumThreads();
	std::vector<std::thread> threads;
	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		threads.emplace_back(std::thread(

		[fun](int t_num,int t_id)
		{
			fun( t_num, t_id);
		},

		num_threads, thread_id));
	}

	for (auto & t : threads)
		t.join();
#endif
}

template<typename TRange>
void ParallelForEach(TRange r, std::function<void(typename TRange::iterator::value_type)> fun)
{
#ifdef _OPENMP

	int num_threads = omp_get_num_procs();

#pragma omp parallel for
	for ( int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		for(auto s:r.Split(num_threads,thread_id))
		{
			fun(s);
		}
	}
#else

	ParallelDo([r,fun](int t_num,int t_id)
	{
		for(auto s:r.Split(t_num,t_id))
		{
			fun(s);
		}
	});
#endif
}

}  // namespace simpla

#endif /* MULTI_THREAD_H_ */
