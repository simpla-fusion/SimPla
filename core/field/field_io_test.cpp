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
#include "domain_dummy.h"

#include "../utilities/log.h"
//#include "../manifold/manifold.h"
//#include "../manifold/domain.h"
//#include "../manifold/topology/structured.h"
//#include "../manifold/geometry/cartesian.h"
//#include "../manifold/diff_scheme/fdm.h"
//#include "../manifold/interpolator/interpolator.h"

#include "../parallel/parallel.h"
#include "../parallel/message_comm.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "../io/data_stream.h"
#include "../physics/constants.h"

using namespace simpla;

int main(int argc, char **argv)
{
	using namespace simpla;

	LOGGER.set_stdout_visable_level(12);

	GLOBAL_COMM.init(argc,argv);
	GLOBAL_DATA_STREAM.cd("field_io_test.h5:/");

	nTuple<Real, 3> xmin = { 0, 0, 0 };

	nTuple<Real, 3> xmax = { 1.0, 2.0, 1.0 };

	nTuple<size_t, 3> dims = { 40, 50, 1 };

//	typedef Manifold<CartesianCoordinates<StructuredMesh> > mesh_type;

	DomainDummy<> domain;
//
//	mesh_type mesh;
//
//	mesh.dimensions(dims);
//	mesh.extents(xmin, xmax);
//
//	mesh.update();

//	auto f0 = make_field<int>(domain);
//	auto f1 = make_field<int>(domain);
//
//	f0.clear();
//	f1.clear();
//
//	for (auto s : mesh.select(VERTEX))
//	{
//		auto idx = (mesh_type::decompact(s) >> mesh_type::MAX_DEPTH_OF_TREE)
//				- mesh.global_begin_;
//
//		f0[s] = idx[0] + (GLOBAL_COMM.get_rank())*100;
//
//		f1[s] = idx[1] + (GLOBAL_COMM.get_rank())*100;
//
//	}
//
//	INFORM << SAVE(f0);
//	INFORM << SAVE(f1);
//	GLOBAL_DATA_STREAM.cd("/d1/");
//	GLOBAL_DATA_STREAM.set_property("Enable Compact Storage" ,true);
//
//	INFORM << simpla::save("f1a", f1);
//	INFORM << simpla::save("f1a", f1);
//	INFORM << simpla::save("f1a", f1);
//	GLOBAL_DATA_STREAM.cd("/d2/");
//	GLOBAL_DATA_STREAM.set_property("Enable Compact Storage" ,false);
//
//	INFORM << simpla::save("f0b", f0, DataStream::SP_RECORD);
//	INFORM << simpla::save("f0b", f0, DataStream::SP_RECORD);
//	INFORM << simpla::save("f0b", f0, DataStream::SP_RECORD);
//
//	INFORM << simpla::save("f1c", f1);
//	INFORM << simpla::save("f1c", f1);
//	INFORM << simpla::save("f1c", f1);
//
//	int cache_depth = 5;
//	GLOBAL_DATA_STREAM.cd("/d3/");
//	GLOBAL_DATA_STREAM.set_property("Cache Depth",cache_depth);
//	for (int i = 0; i < 12; ++i)
//	{
//		INFORM << simpla::save("f1d", f1, DataStream::SP_CACHE);
//		INFORM
//				<< simpla::save("f0d", f0,
//						DataStream::SP_CACHE | DataStream::SP_RECORD);
//	}
//	GLOBAL_DATA_STREAM.command("Flush");
//
//	GLOBAL_DATA_STREAM.cd("/d4/");
//
//	int rank = GLOBAL_COMM.get_rank();
//	std::vector<int> vec(3 * (rank + 1));
//	std::generate(vec.begin(), vec.end(), [rank]()->int
//	{	return ( (rank+1)*1000);});
//	size_t size = vec.size();
//	INFORM << GLOBAL_DATA_STREAM.write("data",&vec[0],DataType::create<int>(),1,nullptr,&size,nullptr,nullptr,nullptr,nullptr ,DataStream::SP_UNORDER);
//
//	auto fv = mesh.make_field<EDGE, Real>();
//
//	fv.clear();
//
//	for (auto s : mesh.select(EDGE))
//	{
//		auto x = mesh.get_coordinates(s);
//
//		fv[s] = std::sin(
//				x[0] * TWOPI / (xmax[0] - xmin[0])
//						+ 2.0 * x[1] * TWOPI / (xmax[1] - xmin[1]));
//
//	}
//	INFORM << simpla::save("fv", fv, DataStream::SP_RECORD);
//	INFORM << simpla::save("fv", fv, DataStream::SP_RECORD);
//	INFORM << simpla::save("fv", fv, DataStream::SP_RECORD);
}

