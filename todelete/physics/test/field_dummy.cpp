/**
 * @file field_dummy.cpp
 *
 *  Created on: 2015-1-27
 *      Author: salmon
 */
#include "../../mesh/Chart.h"
#include "../../toolbox/PrettyStream.h"
#include "../../manifold/pre_define/PreDefine.h"
#include "simpla/mesh/Atlas.h"

#include "simpla/physics/Field.h"


using namespace simpla;


int main(int argc, char **argv)
{

    typedef manifold::CartesianManifold mesh_type;
    typedef field_t<double, mesh_type, mesh::VERTEX> field_type;
    std::cout << traits::type_id<field_type>::name() << std::endl;

    mesh::MeshAtlas m;
    std::shared_ptr<mesh_type> mesh;


    mesh = m.add<mesh_type>();

    mesh->dimensions(index_tuple {10, 10, 10});

    mesh->box(box_type{{0, 0, 0},
                       {1, 1, 1}});

    mesh->deploy();

    print(std::cout, *mesh) << std::endl;

    auto block_id = mesh->uuid();

    auto f = m.make_attribute<field_type>();

    f.view(block_id);

    f = 0;
    f = f * 2;

    std::cout << mesh_type::class_name() << std::endl;

//    get_mesh::DummyMesh d_mesh;
//
//    d_mesh.m_entities_.insert(1);
//    d_mesh.m_entities_.insert(3);
//    d_mesh.m_entities_.insert(4);
//    d_mesh.m_entities_.insert(5);
//
//    std::cout << d_mesh.m_entities_.size() << std::endl;
//
//    for (auto const &v:d_mesh.entity_id_range(get_mesh::VERTEX))
//    {
//        std::cout << v << std::endl;
//    }
    INFORM << DONE << std::endl;
}
