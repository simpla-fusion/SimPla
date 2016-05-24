/*
 * field_dummy.cpp
 *
 *  Created on: 2015-1-27
 *      Author: salmon
 */
#include "../../mesh/MeshBase.h"
#include "../../mesh/MeshAtlas.h"


#include "../FieldBase.h"

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

    std::cout << f->attribute()->get_class_name() << std::endl;

}
