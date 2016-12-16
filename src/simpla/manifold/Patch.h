//
// Created by salmon on 16-12-8.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <map>
#include <simpla/mesh/DataBlock.h>

namespace simpla { namespace mesh
{
struct DataBlock;
struct Chart;

class Patch
{
public:
    std::shared_ptr<Chart> const &mesh() const { return m_mesh_; }

    void mesh(std::shared_ptr<Chart> const &m) { m_mesh_ = m; }

    std::shared_ptr<DataBlock> &
    data(std::string const &id, std::shared_ptr<DataBlock> const &p = (nullptr))
    {
        return m_data_.emplace(id, p).first->second;
    }

    std::shared_ptr<DataBlock>
    data(std::string const &id) const
    {
        auto it = m_data_.find(id);
        if (it != m_data_.end()) { return it->second; } else { return std::shared_ptr<DataBlock>(nullptr); }
    }

    template<typename U>
    U const *data_as(std::string const &n) const
    {
        auto d = data(n);
        ASSERT(d->is_a(typeid(U)));
        return static_cast<U *>(d.get());
    }

    template<typename U>
    U *data_as(std::string const &n, std::shared_ptr<U> const &p = nullptr)
    {
        auto &d = data(n, p);
        ASSERT(d->is_a(typeid(U)));
        return static_cast<U *>(d.get());
    }

    template<typename U>
    U const *mesh_as() const
    {
        ASSERT(m_mesh_->is_a(typeid(U)));
        return static_cast<U *>(m_mesh_.get());
    }

private:
    std::shared_ptr<Chart> m_mesh_;
    std::map<std::string, std::shared_ptr<DataBlock> > m_data_;
};


class PatchCollection
{
public:
private:
    std::map<id_type, std::shared_ptr<Patch> > m_patches_;
};

}}//namespace simpla { namespace mesh

#endif //SIMPLA_PATCH_H
