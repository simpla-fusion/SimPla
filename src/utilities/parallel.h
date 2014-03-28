/*
 * parallel.h
 *
 *  Created on: 2014年3月27日
 *      Author: salmon
 */

#ifndef PARALLEL_H_
#define PARALLEL_H_
#include <thread>
namespace simpla
{

inline void ParallelDo(std::function<void(int, int)> fun)
{
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

}

template<typename TRange>
TRange Split(TRange range, int num, int sub)
{
	return range.Split(num, sub);
}
template<typename TRange>
void ParallelForEach(TRange range, std::function<void(typename TRange::value_type)> fun)
{
	ParallelDo([range,fun](int t_num,int t_id)
	{
		for(auto s:Split(range,t_num,t_id))
		{
			fun(s);
		}
	});

}

}  // namespace simpla

#endif /* PARALLEL_H_ */
