/**
 * @file MeshAtlas.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESH_MESHATLAS_H
#define SIMPLA_MESH_MESHATLAS_H

#include <type_traits>
#include "Mesh.h"
#include "MeshBase.h"
#include "../gtl/Log.h"
#include "MeshWorker.h"

namespace simpla { namespace mesh
{


/**
 *  manager of mesh blocks
 *  - adjacencies (graph, ?r-tree)
 *  - refine and coarsen
 *  - coordinates map on overlap region
 */
class MeshAtlas
{

public:
    enum MESH_STATUS
    {
        LOCAL = 1, // 001
        ADJACENT = 2, // 010

    };

    template<typename T, typename ...Args>
    T make_attribute(Args &&...args) const
    {
        static_assert(std::is_base_of<MeshAttributeBase, typename T::attribute_type>::value,
                      " T can not be converted to MeshAttributeBase!!");
        return T(*this, std::forward<Args>(args)...);
    }

    template<typename T>
    T make_attribute() const
    {
        static_assert(std::is_base_of<MeshAttributeBase, typename T::attribute_type>::value,
                      " T can not be converted to MeshAttributeBase!!");
        return T(*this);
    }

    std::vector<MeshBlockId> adjacent_blocks(MeshBlockId const &id, int inc_level = 0, int status_flag = 0);

    std::vector<MeshBlockId> find(int level = 0, int status_flag = 0);

    int count(int level = 0, int status_flag = 0);

    bool has(MeshBlockId const &id) const { return m_mesh_atlas_.find(id) != m_mesh_atlas_.end(); }

    /**return the id of  root block*/
    MeshBlockId root() const { return m_mesh_atlas_.begin()->first; }

    void set(MeshBlockId const &id, std::shared_ptr<MeshBase> ptr) { m_mesh_atlas_[id].swap(ptr); }

    std::shared_ptr<MeshBase> at(MeshBlockId const &id) const
    {
        return m_mesh_atlas_.at(id);
//        if (res != m_mesh_atlas_.end())
//        {
//            return *res;
//        }
//        else
//        {
//            return std::shared_ptr<MeshBase>(nullptr);
//        }
    }

    template<typename TM>
    TM *at(MeshBlockId const &id) const
    {


        auto res = m_mesh_atlas_.at(id);


        static_assert(std::is_base_of<MeshBase, TM>::value, "TM is not derived from MeshBase!!");


        if (!res->template is_a<TM>()) { BAD_CAST << ("illegal mesh type conversion!") << std::endl; }


        auto ptr = std::dynamic_pointer_cast<TM>(res).get();


        return ptr;
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

    template<typename TM, typename ...Args>
    std::shared_ptr<TM> add(Args &&...args)
    {
        auto ptr = std::make_shared<TM>(std::forward<Args>(args)...);
        MeshBlockId uuid = ptr->uuid();
        m_mesh_atlas_.emplace(std::make_pair(uuid, std::dynamic_pointer_cast<MeshBase>(ptr)));

        return ptr;
    };

    void remove(MeshBlockId const &id)
    {
        m_mesh_atlas_.erase(id);
        while (count(m_max_level_ - 1) == 0)
        {
            --m_max_level_;
            if (m_max_level_ == 0)break;
        }
    }

    void setup() { };

    void teardown() { };

    virtual io::IOStream &check_point(io::IOStream &os) const { return os; };

    virtual io::IOStream &save(io::IOStream &os) const { return os; };

    virtual io::IOStream &load(io::IOStream &is) const { return is; };

    void next_step(Real dt) { };


    template<typename TSolver>
    std::shared_ptr<TSolver> register_solver(MeshBlockId const &w_id)
    {
        static_assert(std::is_base_of<MeshWorker, TSolver>::value, "TSovler is not derived from MeshWorker.");
        auto res = std::make_shared<TSolver>(*this);
        m_workers_.emplace(std::make_pair(w_id, std::dynamic_pointer_cast<MeshWorker>(res)));
        return res;
    }

    template<typename TSolver>
    std::shared_ptr<TSolver> get_worker(MeshBlockId const &w_id) const
    {
        assert(m_workers_.at(w_id)->template is_a<TSolver>());
        return std::dynamic_pointer_cast<TSolver>(m_workers_.at(w_id));
    }

private:
    int m_level_ratio_;

    MeshBlockId m_root_;

    std::map<MeshBlockId, std::shared_ptr<MeshBase> > m_mesh_atlas_;

    std::map<MeshBlockId, std::shared_ptr<MeshWorker> > m_workers_;

    int m_max_level_ = 1;
};

}}//namespace simpla{namespace mesh{

#endif //SIMPLA_MESH_MESHATLAS_H
