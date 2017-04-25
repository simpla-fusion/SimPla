//
// Created by salmon on 17-1-20.
//

#ifndef SIMPLA_MESH_ALL_H
#define SIMPLA_MESH_ALL_H

#include "CartesianGeometry.h"
#include "CoRectMesh.h"
#include "CylindricalGeometry.h"
#include "Mesh.h"
#include "SMesh.h"
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {

typedef Mesh<CartesianGeometry, CoRectMesh> CartesianCoRectMesh;
typedef Mesh<CylindricalGeometry, SMesh> CylindricalSMesh;
}
}
#endif  // SIMPLA_MESH_ALL_H
