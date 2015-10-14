/**
 * @file demo_generalField_op.cpp
 *
 * @date 2015-3-4
 * @author salmon
 */

#include <stddef.h>
#include <memory>

#include "../../core/application/application.h"
#include "../../core/application/use_case.h"
#include "../../core/io/io.h"
#include "../../core/mesh/mesh.h"
#include "manifold"
#include "manifold"
#include "manifold"
#include "manifold"
#include "manifold"
#include "../../core/parallel/mpi_comm.h"
#include "../../core/utilities/log.h"
#include "../../core/field/field_sequence.h"
#include "manifold"

using namespace simpla;

USE_CASE(generalField_op)
{

	typedef CartesianCoordinate<RectMesh> mesh_type;
//	typedef SimpleMesh mesh_type;

	typedef typename mesh_type::coordinate_tuple coordinate_tuple;
	typedef typename mesh_type::index_tuple index_tuple;

	index_tuple dims =
	{ 16, 16, 16 };
	index_tuple ghost_width =
	{ 2, 2, 0 };
	coordinate_tuple xmin =
	{ 0, 0, 0 };
	coordinate_tuple xmax =
	{ 1, 1, 1 };
	auto mesh = make_mesh<mesh_type>();
	mesh->dimensions(dims);
	mesh->extents(xmin, xmax);
	mesh->ghost_width(ghost_width);
	mesh->deploy();

	auto f1 = make_form<EDGE, double>(mesh);
	auto f2 = make_form<FACE, double>(mesh);
	auto m_range = mesh->range();

	f1.deploy();
	f2.deploy();
	size_t count = 0;

	for (auto const & key : m_range)
	{
		f1[key] = count + (GLOBAL_COMM.process_num()+1) *100;
		++count;
	}

	f2.fill(GLOBAL_COMM.process_num() );

	cd("/Output/");

	VERBOSE << SAVE(f1) << std::endl;
	VERBOSE << SAVE(f2) << std::endl;
	f1.sync();
	f2.sync();
	f2.sync();
	f1.sync();
	f1.wait();

	cd("/Output2/");
	VERBOSE << SAVE(f1) << std::endl;
	VERBOSE << SAVE(f2) << std::endl;
} // USE_CASE(generalField_op)

