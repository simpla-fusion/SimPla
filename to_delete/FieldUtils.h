//
// Created by salmon on 16-11-16.
//

#ifndef SIMPLA_FIELDUTILS_H
#define SIMPLA_FIELDUTILS_H

#include <simpla/SIMPLA_config.h>

#include <type_traits>
#include <cassert>

#include <simpla/toolbox/type_traits.h>

#include <simpla/mesh/AttributeView.h>
#include <simpla/mesh/Worker.h>
#include <simpla/mesh/MeshCommon.h>

#include <simpla/toolbox/BoxUtility.h>

#include "Field.h"

namespace simpla { namespace physics
{
template<typename ...> class Field;


template<typename TV, typename TManifold, size_t I, typename TFun>
void assign(Field<TV, TManifold, int_const<I>> &f, box_type const &s_b, TFun const &fun)
{
    auto const &m = f.mesh();
    toolbox::intersection(f.mesh().box(), s_b);

};
}}
#endif //SIMPLA_FIELDUTILS_H
