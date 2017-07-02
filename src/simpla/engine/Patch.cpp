//
// Created by salmon on 17-2-22.
//
#include "Patch.h"
#include <simpla/geometry/GeoObject.h>
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <map>
#include "../../../cmake-build-debug/include/simpla/SIMPLA_config.h"
#include "MeshBlock.h"

namespace simpla {
namespace engine {
struct Patch::pimpl_s {
    id_type m_id_ = NULL_ID;
    MeshBlock m_block_;
    std::map<id_type, DataPack> m_data_;
    std::map<std::string, EntityRange> m_range_;
    static boost::uuids::random_generator m_gen_;
    static boost::hash<boost::uuids::uuid> m_hasher_;
};
boost::uuids::random_generator Patch::pimpl_s::m_gen_;
boost::hash<boost::uuids::uuid> Patch::pimpl_s::m_hasher_;
Patch::Patch(id_type id) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_id_ = id != NULL_ID ? id : m_pimpl_->m_hasher_(m_pimpl_->m_gen_());
}
Patch::~Patch() {}
Patch::Patch(this_type const &other) { UNIMPLEMENTED; }
Patch::Patch(this_type &&other) : m_pimpl_(other.m_pimpl_.get()) { other.m_pimpl_.reset(); }

Patch &Patch::operator=(Patch const &other) {
    Patch(other).swap(*this);
    return *this;
}
Patch &Patch::operator=(Patch &&other) {
    Patch(other).swap(*this);
    return *this;
}
void Patch::swap(Patch &other) { std::swap(m_pimpl_, other.m_pimpl_); }

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