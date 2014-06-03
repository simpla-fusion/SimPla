/*
 * field_io_test.cpp
 *
 *  Created on: 2014年5月12日
 *      Author: salmon
 */

#include "fetl.h"
#include "save_field.h"

#include "../utilities/log.h"

#include "../mesh/octree_forest.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_euclidean.h"

#include "../parallel/parallel.h"

int main(int argc, char **argv)
{

#ifdef USE_PARALLEL_IO
	GLOBAL_COMM.Init(argc,argv);
#endif

	using namespace simpla;

	nTuple<3, Real> xmin =
	{	-1.0, -1.0, -1.0};

	nTuple<3, Real> xmax =
	{	1.0, 1.0, 1.0};

	nTuple<3, size_t> dims =
	{	32, 20, 5};

	typedef Mesh< EuclideanGeometry<OcForest>> mesh_type;

	mesh_type mesh;

	mesh.SetExtents(dims,xmin, xmax);

#if USE_PARALLEL_IO
	mesh.Decompose(GLOBAL_COMM.GetSize(), GLOBAL_COMM.GetRank());
#endif

	Field<mesh_type, VERTEX, Real> f(mesh);

	f.Fill(1234);

#if USE_PARALLEL_IO
	f.Fill(GLOBAL_COMM.GetRank()+100);
#endif

	GLOBAL_DATA_STREAM.OpenFile("FetlTest");
	GLOBAL_DATA_STREAM.OpenGroup("/t1");
	LOGGER << SAVE(f);
	GLOBAL_DATA_STREAM.OpenGroup("/t2");
	GLOBAL_DATA_STREAM.EnableCompactStorable( );
	LOGGER << SAVE(f);
	LOGGER << SAVE(f);
	LOGGER << endl;

}

