/*
 * field_dummy.cpp
 *
 *  Created on: 2015-1-27
 *      Author: salmon
 */

#include "../../mesh/MeshAtlas.h"


#include "../FieldBase.h"

using namespace simpla;
using namespace simpla::mesh;
struct DummyMesh : public MeshBase
{

};

int main(int argc, char **argv)
{
    mesh::MeshAtlas m;

    typedef FieldAttr<double, DummyMesh, mesh::VERTEX> field_type;

    auto f = m.attribute<field_type>();

}
