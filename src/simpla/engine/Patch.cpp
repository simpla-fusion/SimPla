//
// Created by salmon on 17-2-22.
//

#include "Patch.h"
#include <simpla/SIMPLA_config.h>
#include <simpla/geometry/GeoObject.h>
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <map>
#include <memory>
#include "MeshBlock.h"

namespace simpla {
namespace engine {
struct Patch::pimpl_s {
    id_type m_id_ = NULL_ID;
    MeshBlock m_block_;
    std::map<id_type, DataPack> m_data_;
    std::shared_ptr<std::map<std::string, EntityRange>> m_ranges_;
};

Patch::Patch(id_type id) : m_pimpl_(new pimpl_s) {
    static boost::uuids::random_generator m_gen_;
    static boost::hash<boost::uuids::uuid> m_hasher_;
    m_pimpl_->m_id_ = id != NULL_ID ? id : m_hasher_(m_gen_());
    m_pimpl_->m_ranges_ = std::make_shared<std::map<std::string, EntityRange>>();
}
Patch::~Patch() {}
Patch::Patch(this_type const &other) : Patch(other.GetId()) {
    MeshBlock(other.GetBlock()).swap(m_pimpl_->m_block_);
    m_pimpl_->m_data_ = other.m_pimpl_->m_data_;
    m_pimpl_->m_ranges_ = other.m_pimpl_->m_ranges_;
}
Patch::Patch(this_type &&other) : m_pimpl_(other.m_pimpl_.get()) { other.m_pimpl_.reset(); }

void Patch::swap(Patch &other) { std::swap(m_pimpl_, other.m_pimpl_); }

Patch &Patch::operator=(Patch const &other) {
    Patch(other).swap(*this);
    return *this;
}
Patch &Patch::operator=(Patch &&other) {
    Patch(other).swap(*this);
    return *this;
}

bool Patch::empty() const { return m_pimpl_->m_id_ == NULL_ID; };

id_type Patch::GetId() const { return m_pimpl_->m_id_; }

void Patch::SetBlock(const MeshBlock &m) { MeshBlock(m).swap(m_pimpl_->m_block_); }

const MeshBlock &Patch::GetBlock() const { return m_pimpl_->m_block_; }

std::map<id_type, DataPack> &Patch::GetAllData() { return m_pimpl_->m_data_; };

void Patch::Push(id_type id, DataPack &&d) { d.swap(m_pimpl_->m_data_[id]); }

DataPack Patch::Pop(id_type const &id) const {
    DataPack res;
    auto it = m_pimpl_->m_data_.find(id);
    if (it != m_pimpl_->m_data_.end()) {
        res.swap(it->second);
        m_pimpl_->m_data_.erase(it);
    }
    return std::move(res);
}
EntityRange Patch::GetRange(const std::string &g) const {
    auto it = m_pimpl_->m_ranges_->find(g);
    if (it != m_pimpl_->m_ranges_->end()) {
        return it->second;
    } else {
        return EntityRange{};
    }
}
EntityRange &Patch::GetRange(const std::string &g) { return (*m_pimpl_->m_ranges_)[g]; }

EntityRange &Patch::AddRange(const std::string &g, EntityRange &&r) {
    EntityRange &res = (*m_pimpl_->m_ranges_)[g];
    res.append(std::forward<EntityRange>(r));
    return res;
}

std::shared_ptr<std::map<std::string, EntityRange>> Patch::GetRanges() { return m_pimpl_->m_ranges_; }
void Patch::SetRanges(std::shared_ptr<std::map<std::string, EntityRange>> const &r) {
    m_pimpl_->m_ranges_->insert(r->begin(), r->end());
}
//
// void Patch::PushRange(std::shared_ptr<std::map<std::string, EntityRange>> const &r) { m_pimpl_->m_range_ = r; };
// std::shared_ptr<std::map<std::string, EntityRange>> Patch::PopRange() { return m_pimpl_->m_range_; };

}  // namespace engine {
}  // namespace simpla {