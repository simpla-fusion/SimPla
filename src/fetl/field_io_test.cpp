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
	LOG_STREAM.set_stdout_visable_level(12);

	GLOBAL_COMM.Init(argc,argv);

	using namespace simpla;

	nTuple<3, Real> xmin =
	{	0, 0, 0};

	nTuple<3, Real> xmax =
	{	1.0, 1.0, 1.0};

	nTuple<3, size_t> dims =
	{	10, 16, 1};

	typedef Mesh< CartesianGeometry<UniformArray>,false> mesh_type;

	mesh_type mesh;

	mesh.set_dimensions(dims);
	mesh.set_extents(xmin, xmax);

	mesh.Update();

	auto f0=mesh.template make_field<VERTEX,Real> ();
	auto f1=mesh.template make_field<VERTEX,Real> ();

	f0.clear();
	f1.clear();

	for(auto s: mesh.Select(VERTEX))
	{
		auto idx=(mesh_type::Decompact(s)>>mesh_type::MAX_DEPTH_OF_TREE)-mesh.global_begin_;

		f0[s]=idx[0]+(GLOBAL_COMM.get_rank())*100;

		f1[s]=idx[1]+(GLOBAL_COMM.get_rank())*100;

	}
	GLOBAL_DATA_STREAM.open_file("field_io_test");
	GLOBAL_DATA_STREAM.open_group("/");
	LOGGER << SAVE(f0);
	LOGGER << SAVE(f1);

	GLOBAL_DATA_STREAM.set_property("Enable Compact Storage" ,true);
	LOGGER<<simpla::save("f1a",f1,true);
	LOGGER<<simpla::save("f1a",f1,true);
	LOGGER<<simpla::save("f1a",f1,true);
	LOGGER<<simpla::save("f1b",f1,false);
	LOGGER<<simpla::save("f1b",f1,false);
	LOGGER<<simpla::save("f1b",f1,false);
	GLOBAL_DATA_STREAM.set_property("Enable Compact Storage" ,false);
	LOGGER<<simpla::save("f1c",f1 );
	LOGGER<<simpla::save("f1c",f1 );
	LOGGER<<simpla::save("f1c",f1 );

	//	GLOBAL_DATA_STREAM.open_group("/t2");

//	LOGGER << SAVE(f);
//	LOGGER << SAVE(f);
//	LOGGER << endl;
//
	int rank=GLOBAL_COMM.get_rank();
	std::vector<int> vec(12);
	std::generate(vec.begin(), vec.end(),[rank]()->int
			{	return (std::rand()%1000+(rank+1)*1000);});
	LOGGER << GLOBAL_DATA_STREAM.write("data",&vec[0],DataType::create<int>(),vec.size());

}

