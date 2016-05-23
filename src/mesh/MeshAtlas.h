/**
 * @file MeshAtlas.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESH_MESHATLAS_H
#define SIMPLA_MESH_MESHATLAS_H

#include <boost/graph/adjacency_list.hpp>
#include "Mesh.h"
#include "../gtl/Log.h"

namespace simpla { namespace mesh
{

enum MESH_STATUS
{
    LOCAL = 1, // 001
    ADJACENT = 2, // 010

};

class MeshAtlas
{
    int m_level_ratio_;
    mesh_id m_root_;

    std::map<mesh_id, std::shared_ptr<MeshBase> > m_mesh_atlas_;
    int m_max_level_ = 1;
public:

    std::vector<mesh_id> adjacent_blocks(mesh_id const &id, int inc_level = 0, int status_flag = 0);

    std::vector<mesh_id> find(int level = 0, int status_flag = 0);

    int count(int level = 0, int status_flag = 0);


    void set(mesh_id const &id, std::shared_ptr<MeshBase> ptr)
    {
        m_mesh_atlas_[id].swap(ptr);
    };

    std::shared_ptr<MeshBase> get(mesh_id const &id) const
    {
        auto res = m_mesh_atlas_.find(id);
        if (res != m_mesh_atlas_.end())
        {
            return *res;
        }
        else
        {
            return std::shared_ptr<MeshBase>(nullptr);
        }
    }

    int level_ratio() const
    {
        return m_level_ratio_;
    }

    void level_ratio(int m_level_ratio_)
    {
        MeshAtlas::m_level_ratio_ = m_level_ratio_;
    }

    int max_level() const
    {
        return m_max_level_;
    }


    mesh_id add(box_type const &b, int level = 0)
    {
        m_max_level_ = std::max(m_max_level_, level);
//        mesh_id uuid = get_id();
//        m_mesh_atlas_.emplace(std::make_pair(uuid, ptr));
        UNIMPLEMENTED;
        return 0;
    };

    void remove(mesh_id const &id)
    {
        m_mesh_atlas_.erase(id);
        while (count(m_max_level_ - 1) == 0)
        {
            --m_max_level_;
            if (m_max_level_ == 0)break;
        }
    }


};

}}//namespace simpla{namespace mesh{

#endif //SIMPLA_MESH_MESHATLAS_H
