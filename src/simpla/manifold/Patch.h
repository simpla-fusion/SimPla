//
// Created by salmon on 16-12-8.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <map>

namespace simpla { namespace mesh
{

class Patch
{
public:
    std::shared_ptr<MeshBlock> const &mesh() const { return m_mesh_block_; }

    void set_mesh(std::shared_ptr<MeshBlock> const &m) const { m_mesh_block_ = m; }

    std::shared_ptr<DataBlock> &data(id_type const &id, std::shared_ptr<DataBlock> const &p = (nullptr))
    {
        return m_data_.emplace(id, p).first->second;
    }

    std::shared_ptr<DataBlock> data(id_type const &id) const
    {
        auto it = m_data_.find(id);
        if (it != m_data_.end()) { return it->second; } else { return std::shared_ptr<DataBlock>(nullptr); }
    }


private:
    std::shared_ptr<MeshBlock> m_mesh_block_;
    std::map<id_type, std::shared_ptr<DataBlock> > m_data_;
};


class PatchCollection
{
public:
private:
    std::map<id_type, std::shared_ptr<Patch> > m_patches_;
};

}}//namespace simpla { namespace mesh

#endif //SIMPLA_PATCH_H
