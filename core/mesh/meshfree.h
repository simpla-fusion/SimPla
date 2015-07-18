/**
 * @file meshfree.h
 *
 * @date 2015-6-16
 * @author salmon
 */

#ifndef CORE_MESH_MESHFREE_H_
#define CORE_MESH_MESHFREE_H_

namespace simpla
{

namespace tags
{

struct mesh_free;

}  // namespace tags

template<typename ... > struct Mesh;

/**
 * @ingroup mesh
 */
template<typename CoordinateSystem>
struct Mesh<CoordinateSystem, tags::mesh_free>
{

};
}  // namespace simpla

#endif /* CORE_MESH_MESHFREE_H_ */
