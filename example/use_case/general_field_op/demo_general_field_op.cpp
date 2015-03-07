/**
 * @file demo_general_field_op.cpp
 *
 * @date 2015年3月4日
 * @author salmon
 */

#include <memory>

#include "../../../core/application/application.h"
#include "../../../core/application/use_case.h"
#include "../../../core/field/field.h"
#include "../../../core/field/field_shared_ptr.h"
#include "../../../core/io/io.h"
#include "../../../core/mesh/mesh.h"
#include "../../../core/mesh/simple_mesh.h"
#include "../../../core/mesh/structured/topology/structured.h"
#include "../../../core/parallel/mpi_comm.h"
#include "../../../core/utilities/log.h"

using namespace simpla;

USE_CASE(general_field_op)
{

//	typedef CartesianCoordinates<StructuredMesh> mesh_type;
	typedef SimpleMesh mesh_type;

	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::index_tuple index_tuple;

	index_tuple dims =
	{ 16, 16, 16 };
	index_tuple ghost_width =
	{ 0, 0, 0 };
	coordinates_type xmin =
	{ 0, 0, 0 };
	coordinates_type xmax =
	{ 1, 1, 1 };
	auto mesh = make_mesh<mesh_type>();
	mesh->dimensions(dims);
	mesh->extents(xmin, xmax);
	mesh->ghost_width(ghost_width);
	mesh->deploy();

	auto f1 = make_field<double>(mesh);

	auto m_range = mesh->range();

	f1.deploy();

	size_t count = 0;

//	for (auto const & key : m_range)
//	{
//		f1[key] = count + (GLOBAL_COMM.process_num()+1) *100;
//		++count;
//	}

	f1.fill(GLOBAL_COMM.process_num() );

	cd("/Output/");

	VERBOSE << SAVE(f1) << std::endl;

	f1.sync();
//	f1.sync();
//	f1.sync();
//	f1.sync();
	f1.wait_to_ready();

	cd("/Output2/");
	VERBOSE << SAVE(f1) << std::endl;

} // USE_CASE(general_field_op)

