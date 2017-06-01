//
// Created by salmon on 17-6-1.
//

#include <simpla/mesh/CoRectMesh.h>
#include <simpla/mesh/RectMesh.h>
#include <simpla/mesh/SMesh.h>
#include "EMFluid.h"
#include "ExtraSource.h"

namespace simpla {
using namespace mesh;
REGISTER_CREATOR_TEMPLATE(EMFluid, SMesh)
REGISTER_CREATOR_TEMPLATE(EMFluid, RectMesh)
REGISTER_CREATOR_TEMPLATE(EMFluid, CoRectMesh)
REGISTER_CREATOR_TEMPLATE(ExtraSource, SMesh)
REGISTER_CREATOR_TEMPLATE(ExtraSource, RectMesh)
REGISTER_CREATOR_TEMPLATE(ExtraSource, CoRectMesh)
}