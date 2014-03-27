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
void Cocurrent(std::function<void(int, int)> fun)
{
	const unsigned int num_threads = std::thread::hardware_concurrency();
	std::vector<std::thread> threads;
	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		threads.emplace_back(std::thread(fun, num_threads, thread_id));
	}
	for (auto & t : threads)
		t.join();

}
}  // namespace simpla

#endif /* PARALLEL_H_ */
