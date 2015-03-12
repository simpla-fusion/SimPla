/**
 * \file data_stream_test.cpp
 *
 * \date    2014年7月13日  上午10:09:40 
 * \author salmon
 */

#include <gtest/gtest.h>
#include <iostream>
#include "data_stream.h"

#include "../dataset/dataset.h"
#include "../utilities/utilities.h"
#include "../physics/constants.h"
#include "../parallel/mpi_comm.h"

using namespace simpla;

TEST(datastream,write)
{
	LOGGER.set_MESSAGE_visable_level(12);
	GLOBAL_COMM.init();
	GLOBAL_DATA_STREAM.init();

	DataStream data_stream;

	typedef nTuple<Real, 3> value_type;

	size_t ndims = 2;
	size_t dims[2] = { 100, 100 };

	auto ds = make_dataset<value_type>(ndims, dims);



	value_type* f0 = reinterpret_cast<value_type*>(ds.data.get());
	for (size_t i = 0; i < dims[0]; ++i)
	{
		for (size_t j = 0; j < dims[1]; ++j)
		{
			size_t s = i * dims[1] + j;
//			f0[s] = std::sin(i * TWOPI / static_cast<Real>(dims[0] - 1));

			f0[s][0] = std::sin(i * TWOPI / static_cast<Real>(dims[0] - 1));
			f0[s][1] = std::cos(
					i * TWOPI / static_cast<Real>(dims[0] - 1)
							+ j * TWOPI / static_cast<Real>(dims[1] - 1));

			f0[s][2] = std::sin(j * TWOPI / static_cast<Real>(dims[1] - 1));
		}
	}

	auto t = data_stream.cd("data_stream_test.h5:/what/");

	LOGGER << data_stream.write("f0", ds) << std::endl;
//	LOGGER << data_stream.write("f1", &f0[0], data_type, 2, nullptr, dims);
//
	LOGGER << data_stream.write("f0a", ds, SP_RECORD) << std::endl;
	LOGGER << data_stream.write("f0a", ds, SP_RECORD) << std::endl;
	LOGGER << data_stream.write("f0a", ds, SP_RECORD) << std::endl;
	LOGGER << data_stream.write("/f0a", ds, SP_RECORD) << std::endl;
//
	data_stream.set_attribute("f0a.m", 1.0);
	data_stream.set_attribute("/f0a.q", -1.0);
	data_stream.set_attribute("/what/f0a.name", "ele");

	data_stream.set_attribute("/.name", "ele");

//	data_stream.set_property("Enable Compact Storage", true);
//	LOGGER << data_stream.write("f1b", &f1[0], data_type, 2, nullptr, dims, nullptr, nullptr, nullptr, nullptr,DataStream::SP_APPEND);
//	LOGGER << data_stream.write("f1b", &f1[0], data_type, 2, nullptr, dims, nullptr, nullptr, nullptr, nullptr,DataStream::SP_APPEND);
//	LOGGER << data_stream.write("f1b", &f1[0], data_type, 2, nullptr, dims, nullptr, nullptr, nullptr, nullptr,DataStream::SP_APPEND);
//	LOGGER << data_stream.write("f1b", &f1[0], data_type, 2, nullptr, dims, nullptr, nullptr, nullptr, nullptr,DataStream::SP_APPEND);
//	data_stream.set_property("Enable Compact Storage", false);

}

