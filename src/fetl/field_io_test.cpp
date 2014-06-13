/*
 * field_io_test.cpp
 *
 *  Created on: 2014年5月12日
 *      Author: salmon
 */

//#include "fetl.h"
#include "save_field.h"

#include "../utilities/log.h"

#include "../mesh/octree_forest.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_euclidean.h"
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
	{	32, 20, 5};

	typedef Mesh< EuclideanGeometry<OcForest>> mesh_type;

	mesh_type mesh;

	mesh.SetExtents(xmin, xmax,dims);

	mesh.Decompose(GLOBAL_COMM.GetSize(), GLOBAL_COMM.GetRank());

	Field<mesh_type, VERTEX, Real> f(mesh);

	f.Fill(GLOBAL_COMM.GetRank()+100);

	GLOBAL_DATA_STREAM.OpenFile("FetlTest");
	GLOBAL_DATA_STREAM.OpenGroup("/t1");
	LOGGER << SAVE(f);
	GLOBAL_DATA_STREAM.OpenGroup("/t2");
	GLOBAL_DATA_STREAM.EnableCompactStorable( );
	LOGGER << SAVE(f);
	LOGGER << SAVE(f);
	LOGGER << endl;

	int rank=GLOBAL_COMM.GetRank();
	std::vector<int> vec(12);
	std::generate(vec.begin(), vec.end(),[rank]()->int
			{	return (std::rand()%1000+(rank+1)*1000);});
	LOGGER << GLOBAL_DATA_STREAM.UnorderedWrite("data",vec);

}

