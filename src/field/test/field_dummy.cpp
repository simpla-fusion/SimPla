/*
 * field_dummy.cpp
 *
 *  Created on: 2015-1-27
 *      Author: salmon
 */
#include <set>
#include "../../mesh/MeshBase.h"
#include "../../mesh/MeshAtlas.h"
#include "../../mesh/CoRectMesh.h"
#include "../../gtl/PrettyStream.h"
#include "../../gtl/nTupleExt.h"
#include "../Field.h"

using namespace simpla;


int main(int argc, char **argv)
{

    typedef FieldAttr<double, mesh::CoRectMesh, mesh::VERTEX> field_type;
    std::cout << traits::type_id<field_type>::name() << std::endl;

    mesh::MeshAtlas m;
    auto res = m.template add<mesh::CoRectMesh>();

    auto block_id = res.first->uuid();

    auto f = m.make_attribute<field_type>();

    f.view(block_id);

    f = 0;


//    mesh::DummyMesh d_mesh;
//
//    d_mesh.m_entities_.insert(1);
//    d_mesh.m_entities_.insert(3);
//    d_mesh.m_entities_.insert(4);
//    d_mesh.m_entities_.insert(5);
//
//    std::cout << d_mesh.m_entities_.size() << std::endl;
//
//    for (auto const &v:d_mesh.range(mesh::VERTEX))
//    {
//        std::cout << v << std::endl;
//    }
    INFORM << DONE << std::endl;
}
