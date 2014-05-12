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

#if USE_MPI
	MESSAGE_COMM.Init(argc,argv);
#endif

	using namespace simpla;

	nTuple<3, Real> xmin = { -1.0, -1.0, -1.0 };

	nTuple<3, Real> xmax = { 1.0, 1.0, 1.0 };

	nTuple<3, size_t> dims = { 16, 32, 67 };

	typedef RectMesh<OcForest, EuclideanGeometry> mesh_type;

	mesh_type mesh;

	mesh.SetDimensions(dims);

	mesh.SetExtent(xmin, xmax);

	Field<mesh_type, VERTEX, Real> f(mesh);

#if USE_MPI

	f.Fill(MESSAGE_COMM.GetRank());
#else
	f.Fill(1234);
#endif

	GLOBAL_DATA_STREAM.OpenFile("FetlTest");
	GLOBAL_DATA_STREAM.OpenGroup("/t1");
	LOGGER << SAVE(f);
	GLOBAL_DATA_STREAM.OpenGroup("/t2");
	GLOBAL_DATA_STREAM.EnableCompactStorable( );
	LOGGER << SAVE(f);
	LOGGER << endl;

}

