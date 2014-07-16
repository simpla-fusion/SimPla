/**
 * \file data_stream_test.cpp
 *
 * \date    2014年7月13日  上午10:09:40 
 * \author salmon
 */

#include <gtest/gtest.h>
#include <iostream>
#include "data_stream.h"
using namespace simpla;

TEST(datastream,write)
{
	LOG_STREAM.set_stdout_visable_level(12);

	DataStream data_stream;
	static constexpr unsigned int ndims = 2;
	size_t dims[ndims] =
	{	10, 10};
	size_t number = dims[0] * dims[1];
	std::vector<int> f0(number), f1(number);

	for (size_t i = 0; i < dims[0]; ++i)
	{
		for (size_t j = 0; j < dims[1]; ++j)
		{
			size_t s = i * dims[1] + j;
			f0[s] = i + (GLOBAL_COMM.get_rank())*100;

			f1[s] = j + (GLOBAL_COMM.get_rank())*100;
		}
	}

	data_stream.cd("data_stream_test.h5:/");

	LOGGER << data_stream.write("f0", &f0[0], DataType::create<int>(), 2, nullptr, dims);
	LOGGER << data_stream.write("f1", &f0[0], DataType::create<int>(), 2, nullptr, dims);

	data_stream.set_property("Enable Compact Storage", true);
	LOGGER << data_stream.write("f0a", &f0[0], DataType::create<int>(), 2, nullptr, dims);
	LOGGER << data_stream.write("f0a", &f0[0], DataType::create<int>(), 2, nullptr, dims);
	LOGGER << data_stream.write("f0a", &f0[0], DataType::create<int>(), 2, nullptr, dims);
	LOGGER << data_stream.write("f0a", &f0[0], DataType::create<int>(), 2, nullptr, dims);
	data_stream.set_property("Enable Compact Storage", false);

	data_stream.set_property("Enable Compact Storage", true);
	LOGGER << data_stream.write("f1b", &f1[0], DataType::create<int>(), 2, nullptr, dims, nullptr, nullptr, nullptr, nullptr,DataStream::SP_APPEND);
	LOGGER << data_stream.write("f1b", &f1[0], DataType::create<int>(), 2, nullptr, dims, nullptr, nullptr, nullptr, nullptr,DataStream::SP_APPEND);
	LOGGER << data_stream.write("f1b", &f1[0], DataType::create<int>(), 2, nullptr, dims, nullptr, nullptr, nullptr, nullptr,DataStream::SP_APPEND);
	LOGGER << data_stream.write("f1b", &f1[0], DataType::create<int>(), 2, nullptr, dims, nullptr, nullptr, nullptr, nullptr,DataStream::SP_APPEND);
	data_stream.set_property("Enable Compact Storage", false);

}

