/*
 * multi_thread.h
 *
 *  Created on: 2014年5月12日
 *      Author: salmon
 */

#ifndef MULTI_THREAD_H_
#define MULTI_THREAD_H_
#include <vector>
#include <thread>
namespace simpla
{

inline void ParallelDo(std::function<void(int, int)> fun)
{
#ifndef DISABLE_MULTI_THREAD
	const unsigned int num_threads = std::thread::hardware_concurrency();
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
#else
	fun(1, 0);
#endif
}

template<typename TRange>
void ParallelForEach(TRange r, std::function<void(typename TRange::value_type)> fun)
{
	ParallelDo([r,fun](int t_num,int t_id)
	{
		for(auto const & s:r.Split(t_num,t_id))
		{
			fun(s);
		}
	});

}

}  // namespace simpla

#endif /* MULTI_THREAD_H_ */
