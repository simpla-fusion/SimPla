/**
 * @file demo_generalField_op.cpp
 *
 * @date 2015-3-4
 * @author salmon
 */

#include <stddef.h>
#include <memory>

#include "../../core/application/application.h"
#include "../../core/io/io.h"

#include "../../core/manifold/pre_define/riemannian.h"
#include "../../core/parallel/mpi_comm.h"
#include "../../core/gtl/utilities/log.h"
#include "../../core/field/field_dense.h"
#include "../../core/field/field_expression.h"


using namespace simpla;

USE_CASE(general_field_op, "General field operation")
{

    typedef manifold::Riemannian<3> mesh_type;
//	typedef SimpleMesh mesh_type;

    typedef typename mesh_type::point_type coordinate_tuple;
    typedef typename mesh_type::index_tuple index_tuple;

    index_tuple dims = {16, 16, 16};
    index_tuple ghost_width = {2, 2, 0};
    coordinate_tuple xmin = {0, 0, 0};
    coordinate_tuple xmax = {1, 1, 1};

    auto mesh = std::make_shared<mesh_type>();

    mesh->dimensions(dims);
    mesh->extents(xmin, xmax);
//	mesh->ghost_width(ghost_width);
    mesh->deploy();

    CHECK(*mesh);

    CHECK(traits::type_id<mesh_type>::name());


    auto f1 = traits::make_field<EDGE, double>(*mesh);
    auto f2 = traits::make_field<FACE, double>(*mesh);
    auto m_range = mesh->range();

    f1.deploy();
    f2.deploy();

    CHECK(f1.is_valid());
    CHECK(f1.data != nullptr);
    CHECK(f1.datatype.is_valid());
    CHECK(f1.dataspace.is_valid());
    size_t count = 0;

    for (auto const &key : m_range)
    {
        f1[key] = count + (GLOBAL_COMM.process_num() + 1) * 100;
        ++count;
    }

    f2.clear();
    f2 = (GLOBAL_COMM.process_num());

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

