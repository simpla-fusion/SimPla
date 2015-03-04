/**
 * @file demo_general_field_op.cpp
 *
 * @date 2015年3月4日
 * @author salmon
 */

#include "../../../core/application/use_case.h"
#include <memory>
#include <string>
#include "../../../core/mesh/mesh.h"
#include "../../../core/mesh/simple_mesh.h"
#include "../../../core/field/field_shared_ptr.h"
#include "../../../core/field/field.h"

#include "../../../core/io/io.h"
#include "../../../core/parallel/parallel.h"
#include "../../../core/parallel/mpi_update.h"
using namespace simpla;

USE_CASE(general_field_op)
{
	typedef typename SimpleMesh::coordinates_type coordinates_type;
	typedef typename SimpleMesh::index_tuple index_tuple;

	index_tuple dims = { 1, 10, 10 };
	coordinates_type xmin = { 0, 0, 0 }, xmax = { 1, 1, 1 };
	auto mesh = make_mesh<SimpleMesh>();
	mesh->dimensions(dims);
	mesh->extents(xmin, xmax);
	mesh->deploy();

	auto f1 = make_field<double>(mesh);

	f1.fill(GLOBAL_COMM.process_num()+1.345);

	cd("/Output/");

	VERBOSE << SAVE(f1) << std::endl;

} // USE_CASE(general_field_op)

