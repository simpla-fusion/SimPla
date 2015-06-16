/**
 * @file unstructured.h
 *
 * @date 2015年6月16日
 * @author salmon
 */

#ifndef CORE_MESH_UNSTRUCTURED_H_
#define CORE_MESH_UNSTRUCTURED_H_

namespace simpla
{
namespace tags
{
template<typename PrimaryShape> struct unstructured;
}
// namespace tags

template<typename ... > struct Mesh;

/**
 * @ingroup mesh
 */
template<typename CoordinateSystem, typename PrimaryShape>
struct Mesh<CoordinateSystem, tags::unstructured<PrimaryShape> >
{

};
}  // namespace simpla

#endif /* CORE_MESH_UNSTRUCTURED_H_ */
