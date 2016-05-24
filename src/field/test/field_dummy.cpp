/*
 * field_dummy.cpp
 *
 *  Created on: 2015-1-27
 *      Author: salmon
 */
#include "../../mesh/MeshBase.h"
#include "../../mesh/MeshAtlas.h"


#include "../Field.h"

using namespace simpla;


struct DummyMesh : public mesh::MeshBase
{
    SP_OBJECT_HEAD(DummyMesh, mesh::MeshBase);

    mesh::MeshEntityRange range(mesh::MeshEntityType) const { return mesh::MeshEntityRange(); }
};

int main(int argc, char **argv)
{
    mesh::MeshAtlas m;

    typedef FieldAttr<double, DummyMesh, mesh::VERTEX> field_type;

    auto f = m.attribute<field_type>();


    std::cout << traits::type_id<field_type>::name() << std::endl;

    f = 0;

}
