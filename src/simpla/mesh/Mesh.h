//
// Created by salmon on 17-4-24.
//

#ifndef SIMPLA_MESHVIEW_H
#define SIMPLA_MESHVIEW_H

#include <simpla/engine/MeshBase.h>
namespace simpla {
namespace mesh {

template <typename... T>
class Mesh : public engine::MeshBase {
    DECLARE_REGISTER_NAME("Mesh<>");
};

template <typename... T>
bool Mesh<T...>::is_registered = Mesh<T...>::template RegisterCreator<Mesh<T...>>();
}  // namespace mesh {
}  // namespace simpla {

#endif  // SIMPLA_MESHVIEW_H
