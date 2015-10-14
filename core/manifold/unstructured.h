/**
 * @file unstructured.h
 *
 * @date 2015-6-16
 * @author salmon
 */

#ifndef CORE_MESH_UNSTRUCTURED_H_
#define CORE_MESH_UNSTRUCTURED_H_

namespace simpla {
namespace tags {
template<typename PrimaryShape> struct unstructured;
}
// namespace tags

template<typename ...> struct Mesh;

/**
 * @ingroup manifold
 */
template<typename CoordinateSystem, typename PrimaryShape>
struct Mesh<CoordinateSystem, tags::unstructured<PrimaryShape> >
{

};
}  // namespace simpla

#endif /* CORE_MESH_UNSTRUCTURED_H_ */
