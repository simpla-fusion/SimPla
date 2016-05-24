/**
 * @file MeshMap.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESH_MESHMAP_H
#define SIMPLA_MESH_MESHMAP_H

#include "Mesh.h"

namespace simpla { namespace mesh
{
namespace tags
{
struct EntityWiseCopy;
struct PointWiseCopy;
struct Refine;
struct Coarse;
}//namespace tags




template<typename MAP>
class MeshMap
{

    MeshEntityRange m_range1_;
    MeshEntityRange m_range2_;
    MAP m_map_;

    template<typename TF1, typename TF2>
    void map(TF1 const &f1, TF2 *f2)
    {
        auto it2 = m_range2_.begin();
        auto ie2 = m_range2_.end();
        for (; it2 != ie2; ++it2) { (*f2)[*it2] = f1(m_map_(f2->mesh().point(*it2))); }
    };
};

template<>
class MeshMap<tags::EntityWiseCopy>
{

    MeshEntityRange m_range1_;
    MeshEntityRange m_range2_;


    template<typename TF1, typename TF2>
    void map(TF1 const &f1, TF2 *f2)
    {
        auto it1 = m_range1_.begin();
        auto ie1 = m_range1_.end();
        auto it2 = m_range2_.begin();

        for (; it1 != ie1; ++it1, ++it2) { (*f2)[*it2] = f1[*it1]; }

    };


};

}}//namespace simpla{namespace mesh{
#endif //SIMPLA_MESH_MESHMAP_H
