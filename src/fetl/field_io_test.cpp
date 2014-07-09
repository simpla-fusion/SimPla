/*
 * field_io_test.cpp
 *
 *  created on: 2014-5-12
 *      Author: salmon
 */

#include "field.h"
#include "save_field.h"

#include "../utilities/log.h"

#include "../mesh/uniform_array.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_cartesian.h"
#include <iostream>
#include "../parallel/parallel.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "../io/data_stream.h"

using namespace simpla;

int main(int argc, char **argv)
{
	GLOBAL_COMM.Init(argc,argv);

	using namespace simpla;

	nTuple<3, Real> xmin =
	{	-1.0, -1.0, -1.0};

	nTuple<3, Real> xmax =
	{	1.0, 1.0, 1.0};

	nTuple<3, size_t> dims =
	{	20, 10, 1};

	typedef Mesh< CartesianGeometry<UniformArray>,false> mesh_type;

	mesh_type mesh;

	mesh.set_dimensions(dims);
	mesh.set_extents(xmin, xmax);

	mesh.Update();

	mesh.Decompose(GLOBAL_COMM.get_size(), GLOBAL_COMM.get_rank());

	auto f=mesh.template make_field< 0,Real> ();

	f.Fill(GLOBAL_COMM.get_rank()+100);

	GLOBAL_DATA_STREAM.OpenFile("FetlTest");
	GLOBAL_DATA_STREAM.OpenGroup("/t1");
	LOGGER << SAVE(f);
	GLOBAL_DATA_STREAM.OpenGroup("/t2");
	GLOBAL_DATA_STREAM.EnableCompactStorable( );
	LOGGER << SAVE(f);
	LOGGER << SAVE(f);
	LOGGER << endl;

	int rank=GLOBAL_COMM.get_rank();
	std::vector<int> vec(12);
	std::generate(vec.begin(), vec.end(),[rank]()->int
			{	return (std::rand()%1000+(rank+1)*1000);});
	LOGGER << GLOBAL_DATA_STREAM.UnorderedWrite("data",vec);

}

