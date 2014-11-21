/**
 * \file data_stream_test.cpp
 *
 * \date    2014年7月13日  上午10:09:40 
 * \author salmon
 */

#include <gtest/gtest.h>
#include <iostream>
#include "data_stream.h"

#include "../parallel/mpi_comm.h"
#include "../utilities/log.h"
#include "../utilities/ntuple.h"
#include "../physics/constants.h"

using namespace simpla;

TEST(datastream,write)
{
	LOGGER.set_stdout_visable_level(12);
	GLOBAL_COMM.init();

	DataStream data_stream;
	static constexpr unsigned int ndims = 2;
	size_t dims[ndims] = { 100, 100 };
	size_t number = dims[0] * dims[1];
	std::vector<nTuple<Real, 3>> f0(number);

	for (size_t i = 0; i < dims[0]; ++i)
	{
		for (size_t j = 0; j < dims[1]; ++j)
		{
			size_t s = i * dims[1] + j;
			f0[s][0] = std::sin(i * TWOPI / static_cast<Real>(dims[0] - 1));

			f0[s][1] = std::cos(
					i * TWOPI / static_cast<Real>(dims[0] - 1)
							+ j * TWOPI / static_cast<Real>(dims[1] - 1));

			f0[s][2] = std::sin(j * TWOPI / static_cast<Real>(dims[1] - 1));
		}
	}

	data_stream.cd("data_stream_test.h5:/what");

	auto data_type = DataType::create<nTuple<Real, 3>>();

	LOGGER << data_stream.write("f0", &f0[0], data_type, 2, nullptr, dims);
//	LOGGER << data_stream.write("f1", &f0[0], data_type, 2, nullptr, dims);
//
	data_stream.set_property("Force Record Storage", true);
	LOGGER << data_stream.write("f0a", &f0[0], data_type, 2, nullptr, dims);
	LOGGER << data_stream.write("f0a", &f0[0], data_type, 2, nullptr, dims);
	LOGGER << data_stream.write("f0a", &f0[0], data_type, 2, nullptr, dims);
	LOGGER << data_stream.write("/f0a", &f0[0], data_type, 2, nullptr, dims);
	data_stream.set_property("Force Record Storage", false);

	data_stream.set_attribute("f0a.m", 1.0);
	data_stream.set_attribute("/f0a.q", -1.0);
	data_stream.set_attribute("/what/f0a.name", "ele");

	data_stream.set_attribute("/.gname", "ele");

//	data_stream.set_property("Enable Compact Storage", true);
//	LOGGER << data_stream.write("f1b", &f1[0], data_type, 2, nullptr, dims, nullptr, nullptr, nullptr, nullptr,DataStream::SP_APPEND);
//	LOGGER << data_stream.write("f1b", &f1[0], data_type, 2, nullptr, dims, nullptr, nullptr, nullptr, nullptr,DataStream::SP_APPEND);
//	LOGGER << data_stream.write("f1b", &f1[0], data_type, 2, nullptr, dims, nullptr, nullptr, nullptr, nullptr,DataStream::SP_APPEND);
//	LOGGER << data_stream.write("f1b", &f1[0], data_type, 2, nullptr, dims, nullptr, nullptr, nullptr, nullptr,DataStream::SP_APPEND);
//	data_stream.set_property("Enable Compact Storage", false);

}

