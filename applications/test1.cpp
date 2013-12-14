/*
 * test1.cpp
 *
 *  Created on: 2013年12月11日
 *      Author: salmon
 */

#include <chrono>
#include <ratio>
#include <thread>
#include <vector>

#include "../src/utilities/log.h"

int main()
{
	const unsigned int num_threads = std::thread::hardware_concurrency();

	CHECK(num_threads);
	std::vector<std::thread> threads(num_threads);

	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		CHECK(thread_id);
		threads[thread_id] =

		std::thread(

		[&,thread_id]()
		{
			CHECK(std::this_thread::get_id())<<"["<< thread_id <<"]";
		}

		);
	}

	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{

		threads[thread_id].join();
	}
}

