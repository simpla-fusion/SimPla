/*
 * memory_pool_test.cpp
 *
 *  Created on: 2014-11-14
 *      Author: salmon
 */

#include <iostream>
#include <iomanip>
#include "memory_pool.h"
#include "log.h"
#include "../io/data_stream.h"
#include "../parallel/mpi_comm.h"

using namespace simpla;

/// TODO need a multithead test of memory pool

int main(int argc, char **argv)
{
	LOGGER.init(argc, argv);
	GLOBAL_COMM.init(argc,argv);
	GLOBAL_DATA_STREAM.init(argc,argv);

//	std::shared_ptr<double> p[10];
//
//	SingletonHolder<MemoryPool>::instance().max_size(4000);
//	for (int i = 0; i < 10; ++i)
//	{
//		p[i] = sp_make_shared_array<double>(100);
//
//		std::cout << SingletonHolder<MemoryPool>::instance().size()
//				<< std::endl;
//	}
//
//	for (int i = 0; i < 10; ++i)
//	{
//		p[i].reset();
//
//		std::cout << SingletonHolder<MemoryPool>::instance().size()
//				<< std::endl;
//	}
//
//	std::cout << "  " << std::setw(25) << std::left << "hello" << " "
//			<< std::left << "world!" << std::endl;

}
