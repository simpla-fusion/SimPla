/*
 * field_io_test.cpp
 *
 *  created on: 2014-5-12
 *      Author: salmon
 */

#include <iostream>
#include <cmath>
#include "field.h"
#include "save_field.h"

#include "../utilities/log.h"

#include "../mesh/uniform_array.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_cartesian.h"

#include "../parallel/parallel.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "../io/data_stream.h"
#include "../physics/constants.h"

using namespace simpla;

int main(int argc, char **argv)
{
	using namespace simpla;

	LOG_STREAM.set_stdout_visable_level(12);

	GLOBAL_COMM.init(argc,argv);
	GLOBAL_DATA_STREAM.cd("field_io_test.h5:/");

	nTuple<3, Real> xmin = { 0, 0, 0 };

	nTuple<3, Real> xmax = { 1.0, 2.0, 1.0 };

	nTuple<3, size_t> dims = { 40, 50, 1 };

	typedef Mesh<CartesianGeometry<UniformArray>, false> mesh_type;

	mesh_type mesh;

	mesh.set_dimensions(dims);
	mesh.set_extents(xmin, xmax);

	mesh.Update();

	auto f0 = mesh.template make_field<VERTEX, int>();
	auto f1 = mesh.template make_field<VERTEX, int>();

	f0.clear();
	f1.clear();

	for (auto s : mesh.Select(VERTEX))
	{
		auto idx = (mesh_type::Decompact(s) >> mesh_type::MAX_DEPTH_OF_TREE) - mesh.global_begin_;

		f0[s] = idx[0] + (GLOBAL_COMM.get_rank())*100;

		f1[s] = idx[1] + (GLOBAL_COMM.get_rank())*100;

	}

	INFORM << SAVE(f0);
	INFORM << SAVE(f1);
	GLOBAL_DATA_STREAM.cd("/d1/");
	GLOBAL_DATA_STREAM.set_property("Enable Compact Storage" ,true);

	INFORM << simpla::save("f1a", f1);
	INFORM << simpla::save("f1a", f1);
	INFORM << simpla::save("f1a", f1);
	GLOBAL_DATA_STREAM.cd("/d2//");
	GLOBAL_DATA_STREAM.set_property("Enable Compact Storage" ,false);

	INFORM << simpla::save("f0b", f0, DataStream::SP_RECORD);
	INFORM << simpla::save("f0b", f0, DataStream::SP_RECORD);
	INFORM << simpla::save("f0b", f0, DataStream::SP_RECORD);

	INFORM << simpla::save("f1c", f1);
	INFORM << simpla::save("f1c", f1);
	INFORM << simpla::save("f1c", f1);

	int cache_depth = 5;
	GLOBAL_DATA_STREAM.cd("/d3/");
	GLOBAL_DATA_STREAM.set_property("Cache Depth",cache_depth);
	for (int i = 0; i < 12; ++i)
	{
		INFORM << simpla::save("f1d", f1, DataStream::SP_CACHE);
		INFORM << simpla::save("f0d", f0, DataStream::SP_CACHE | DataStream::SP_RECORD);
	}
	GLOBAL_DATA_STREAM.command("Flush");

	GLOBAL_DATA_STREAM.cd("/d4/");

	int rank = GLOBAL_COMM.get_rank();
	std::vector<int> vec(3 * (rank + 1));
	std::generate(vec.begin(), vec.end(), [rank]()->int
	{	return ( (rank+1)*1000);});
	size_t size = vec.size();
	INFORM << GLOBAL_DATA_STREAM.write("data",&vec[0],DataType::create<int>(),1,nullptr,&size,nullptr,nullptr,nullptr,nullptr ,DataStream::SP_UNORDER);

	auto fv = mesh.template make_field<EDGE, Real>();

	fv.clear();

	for (auto s : mesh.Select(EDGE))
	{
		auto x = mesh.get_coordinates(s);

		fv[s] = std::sin(x[0] * TWOPI / (xmax[0] - xmin[0]) + 2.0 * x[1] * TWOPI / (xmax[1] - xmin[1]));

	}
	INFORM << simpla::save("fv", fv, DataStream::SP_RECORD);
	INFORM << simpla::save("fv", fv, DataStream::SP_RECORD);
	INFORM << simpla::save("fv", fv, DataStream::SP_RECORD);
}

