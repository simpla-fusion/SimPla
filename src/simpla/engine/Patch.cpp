//
// Created by salmon on 17-2-22.
//
#include "Patch.h"
#include <simpla/geometry/GeoObject.h>
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <map>
#include "MeshBlock.h"
namespace simpla {
namespace engine {
struct Patch::pimpl_s {
    id_type m_id_ = NULL_ID;
    std::shared_ptr<Chart> m_chart_ = nullptr;
    std::shared_ptr<MeshBlock> m_block_ = nullptr;
    std::map<id_type, std::shared_ptr<data::DataBlock>> m_data_;
    std::shared_ptr<std::map<std::string, EntityRange>> m_range_;
    static boost::uuids::random_generator m_gen_;
    static boost::hash<boost::uuids::uuid> m_hasher_;
};
boost::uuids::random_generator Patch::pimpl_s::m_gen_;
boost::hash<boost::uuids::uuid> Patch::pimpl_s::m_hasher_;
Patch::Patch(id_type id) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_id_ = id != NULL_ID ? id : m_pimpl_->m_hasher_(m_pimpl_->m_gen_());
}
Patch::~Patch() {}

id_type Patch::GetId() const { return m_pimpl_->m_id_; }

void Patch::SetBlock(std::shared_ptr<MeshBlock> const &m) { m_pimpl_->m_block_ = m; }
std::shared_ptr<MeshBlock> Patch::GetBlock() const { return m_pimpl_->m_block_; }

void Patch::Merge(Patch &other) {
    if (other.m_pimpl_->m_chart_ != nullptr) { m_pimpl_->m_chart_ = other.m_pimpl_->m_chart_; }
    if (other.m_pimpl_->m_block_ != nullptr) { m_pimpl_->m_block_ = other.m_pimpl_->m_block_; }
    if (m_pimpl_->m_id_ == NULL_ID) { m_pimpl_->m_id_ = other.m_pimpl_->m_id_; }
    m_pimpl_->m_data_.insert(other.m_pimpl_->m_data_.begin(), other.m_pimpl_->m_data_.end());
    if (other.m_pimpl_->m_range_ != nullptr)
        m_pimpl_->m_range_->insert(other.m_pimpl_->m_range_->begin(), other.m_pimpl_->m_range_->end());
}

std::map<id_type, std::shared_ptr<data::DataBlock>> &Patch::GetAllData() { return m_pimpl_->m_data_; };

void Patch::Push(id_type const &id, std::shared_ptr<data::DataBlock> const &d) { m_pimpl_->m_data_[id] = d; }
std::shared_ptr<data::DataBlock> Patch::Pop(id_type const &id) const {
    std::shared_ptr<data::DataBlock> res = nullptr;
    auto it = m_pimpl_->m_data_.find(id);
    if (it != m_pimpl_->m_data_.end()) { res = it->second; }
    return res;
}
// EntityRange Patch::GetRange(const std::string &g) const {
//    if (m_pimpl_->m_range_ != nullptr) {
//        auto it = m_pimpl_->m_range_->find(g);
//        if (it != m_pimpl_->m_range_->end()) { return it->second; }
//    }
//    return EntityRange{};
//}
//
// void Patch::SetRange(const std::string &g, EntityRange const &r) {
//    if (m_pimpl_->m_range_ == nullptr) { m_pimpl_->m_range_ = std::make_shared<std::map<std::string, EntityRange>>();
//    }
//    (*m_pimpl_->m_range_)[g] = r;
//}
//
// void Patch::PushRange(std::shared_ptr<std::map<std::string, EntityRange>> const &r) { m_pimpl_->m_range_ = r; };
// std::shared_ptr<std::map<std::string, EntityRange>> Patch::PopRange() { return m_pimpl_->m_range_; };

}  // namespace engine {
}  // namespace simpla {