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

	index_tuple dims = { 1, 16, 16 };
	index_tuple ghost_width = { 0, 2, 0 };
	coordinates_type xmin = { 0, 0, 0 };
	coordinates_type xmax = { 1, 1, 1 };
	auto mesh = make_mesh<SimpleMesh>();
	mesh->dimensions(dims);
	mesh->extents(xmin, xmax);
	mesh->ghost_width(ghost_width);
	mesh->deploy();

	auto f1 = make_field<double>(mesh);

	f1.fill(GLOBAL_COMM.process_num() );

	cd("/Output/");

	VERBOSE << SAVE(f1) << std::endl;

	std::vector<mpi_send_recv_s> s_r_list;

//	make_send_recv_list(DataType::create<double>(), &s_r_list);

//	sync_update_continue(s_r_list, f1.data().get());

	f1.sync();
	f1.sync();
	f1.sync();
	f1.sync();

	cd("/Output2/");
	VERBOSE << SAVE(f1) << std::endl;

} // USE_CASE(general_field_op)

