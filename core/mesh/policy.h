/**
 * @file policy.h
 *
 * @date 2015-6-16
 * @author salmon
 */

#ifndef CORE_MESH_POLICY_H_
#define CORE_MESH_POLICY_H_

namespace simpla
{

namespace tags
{
struct finite_difference;
struct finite_volume;
struct finite_element;
struct discontinuous_Galerkin;

struct linear;
struct quadric;
struct cubic;
}  // namespace tags

namespace solver
{
template<typename TM, typename TAGS> struct calculate;

template<typename TM, typename TAGS> struct interpolate;

}  // namespace solver
}  // namespace simpla

#endif /* CORE_MESH_POLICY_H_ */
