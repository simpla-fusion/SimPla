//
// Created by salmon on 17-7-8.
//

#ifndef SIMPLA_EBMESH_H
#define SIMPLA_EBMESH_H
namespace simpla {
namespace mesh {
template <typename...>
struct EBMesh;

template <typename TM>
struct EBMesh<TM> {};

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_EBMESH_H
