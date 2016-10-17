//
// Created by salmon on 16-10-17.
//

#ifndef SIMPLA_FIELDEQUATION_H
#define SIMPLA_FIELDEQUATION_H

#include "SIMPLA_config.h"
#include "toolbox/type_traits.h"
#include "toolbox/DataSet.h"
#include "mesh/MeshCommon.h"
#include "mesh/MeshBase.h"
#include "mesh/ModelSelect.h"
#include "mesh/Attribute.h"

namespace simpla
{

template<typename ...> class Field;

template<typename ...U, typename TFun> Field<U...> &
assign(mesh::EntityRange const &r0, Field<U...> &f, TFun const &op,
      CHECK_FUNCTION_SIGNATURE( (typename Field<U...>::field_value_type),
                               TFun(point_type const&, typename Field<U...>::field_value_type const &)))
{
    auto const &m = *f.mesh();
    static const mesh::MeshEntityType IFORM = Field<U...>::iform;
    f.deploy();
    r0.foreach([&](mesh::MeshEntityId const &s)
               {
                   auto x = m.point(s);
                   f[s] = m.template sample<IFORM>(s, op(x, f.gather(x)));
               }
    );


    return f;
}

template<typename ...U, typename TFun> Field<U...> &
assign(mesh::EntityRange const &r0, Field<U...> &f, TFun const &op,
      CHECK_FUNCTION_SIGNATURE(typename Field<U...>::field_value_type,
                               TFun(point_type const&)))
{
    f.deploy();

    auto const &m = *f.mesh();

    static const mesh::MeshEntityType IFORM = Field<U...>::iform;

    r0.foreach([&](mesh::MeshEntityId const &s) { f[s] = m.template sample<IFORM>(s, op(m.point(s))); });


    return f;
}

template<typename ...U, typename TFun> Field<U...> &
assign(mesh::EntityRange const &r0, Field<U...> &f, TFun const &op,
      CHECK_FUNCTION_SIGNATURE(typename Field<U...>::value_type, TFun(mesh::MeshEntityId const &)))
{
    f.deploy();

    auto const &m = *f.mesh();

    static const mesh::MeshEntityType IFORM = Field<U...>::iform;

    r0.foreach([&](mesh::MeshEntityId const &s) { f[s] = op(s); });

    return f;
};

template<typename ...U, typename TFun> Field<U...> &
assign(mesh::EntityRange const &r0, Field<U...> &f, TFun const &op,
      CHECK_FUNCTION_SIGNATURE(typename Field<U...>::value_type, TFun(typename Field<U...>::value_type & )))
{
    f.deploy();

    auto const &m = *f.mesh();

    static const mesh::MeshEntityType IFORM = Field<U...>::iform;

    r0.foreach([&](mesh::MeshEntityId const &s) { op(f[s]); });

    return f;
}

template<typename ...U, typename ...V> Field<U...> &
assign(mesh::EntityRange const &r0, Field<U...> &f, Field<V...> const &g)
{
    f.deploy();

    auto const &m = *f.mesh();

    static const mesh::MeshEntityType IFORM = Field<U...>::iform;

    r0.foreach([&](mesh::MeshEntityId const &s) { f[s] = g[s]; });

    return f;
}


}//namespace simpla
#endif //SIMPLA_FIELDEQUATION_H
