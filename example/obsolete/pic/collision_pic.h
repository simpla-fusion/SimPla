/**
 * @file collision_pic.h
 * @author salmon
 * @date 2015-11-11.
 */

#ifndef SIMPLA_COLLISION_PIC_H
#define SIMPLA_COLLISION_PIC_H


#include <algorithm>
#include <string>
#include <tuple>

#include "../../core/dataset/datatype.h"
#include "../../core/dataset/datatype_ext.h"

#include "../../core/toolbox/ntuple.h"
#include "../../core/sp_def.h"
#include "../../core/toolbox/type_traits.h"

#include "../../core/physics/PhysicalConstants.h"

#include "../../core/particle/ParticleEngine.h"

using namespace simpla;

namespace simpla
{
SP_DEFINE_STRUCT(pic_demo,
                 Vec3, x,
                 Vec3, v,
                 Real, f,
                 Real, w)

}//namespace simpla
#endif //SIMPLA_COLLISION_PIC_H
