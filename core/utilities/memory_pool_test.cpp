/*
 * memory_pool_test.cpp
 *
 *  Created on: 2014年11月14日
 *      Author: salmon
 */

#include <iostream>
#include "memory_pool.h"

using namespace simpla;

/// TODO need a multithead test of memory pool

int main(int argc, char **argv)
{
	std::shared_ptr<double> p[10];

	SingletonHolder<MemoryPool>::instance().max_size(4000);
	for (int i = 0; i < 10; ++i)
	{
		p[i] = sp_make_shared_array<double>(100);

		std::cout << SingletonHolder<MemoryPool>::instance().size()
				<< std::endl;
	}

	for (int i = 0; i < 10; ++i)
	{
		p[i].reset();

		std::cout << SingletonHolder<MemoryPool>::instance().size()
				<< std::endl;
	}
}
