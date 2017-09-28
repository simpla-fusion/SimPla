//
// Created by salmon on 17-3-1.
//
#include "simpla/SIMPLA_config.h"

#include <simpla/data/DataNode.h>
#include <simpla/parallel/MPIComm.h>
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include "simpla/algebra/EntityId.h"

#include "MeshBlock.h"

namespace simpla {
namespace engine {
size_type MeshBlock::m_count_ = 0;

MeshBlock::MeshBlock() {}
MeshBlock::MeshBlock(index_box_type b, int level, size_type local_id)
    : m_level_(level), m_local_id_(local_id == 0 ? (++m_count_) : local_id), m_index_box_(std::move(b)) {
    if (local_id > m_count_) { m_count_ = local_id + 1; }
}

MeshBlock::~MeshBlock() = default;
std::shared_ptr<MeshBlock> MeshBlock::New(std::shared_ptr<simpla::data::DataNode> const &tdb) {
    auto res = std::shared_ptr<MeshBlock>(new MeshBlock);
    res->Deserialize(tdb);
    return res;
}
std::shared_ptr<simpla::data::DataNode> MeshBlock::Serialize() const {
    auto res = data::DataNode::New(data::DataNode::DN_TABLE);
    res->SetValue("LowIndex", std::get<0>(GetIndexBox()));
    res->SetValue("HighIndex", std::get<1>(GetIndexBox()));

    res->SetValue("Level", GetLevel());
    res->SetValue("LocalId", GetLocalID());
    res->SetValue("GUID", GetGUID());
    return res;
}
void MeshBlock::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    std::get<0>(m_index_box_) = cfg->GetValue<index_tuple>("LowIndex", index_tuple{0, 0, 0});
    std::get<1>(m_index_box_) = cfg->GetValue<index_tuple>("HighIndex", index_tuple{1, 1, 1});

    m_level_ = cfg->GetValue<int>("Level", 0);
    m_local_id_ = cfg->GetValue<id_type>("LocalId", m_count_);
}
std::shared_ptr<MeshBlock> MeshBlock::New(index_box_type const &box, int level, size_type local_id) {
    return std::shared_ptr<MeshBlock>(new MeshBlock(box, level, local_id));
};

id_type MeshBlock::GetGUID() const {
    id_type res = m_local_id_;
#ifdef MPI_FOUND
    res = res * GLOBAL_COMM.size() + GLOBAL_COMM.rank();
#endif
    res = res * MAX_LEVEL_NUMBER + m_level_;
    return res;
}

// MeshBlock &MeshBlock::operator=(MeshBlock const &other) {
//    MeshBlock(other).swap(*this);
//    return *this;
//}
// MeshBlock &MeshBlock::operator=(MeshBlock &&other) noexcept {
//    MeshBlock(other).swap(*this);
//    return *this;
//}
// index_tuple MeshBlock::GetGhostWidth() const { return m_ghost_width_; };
// index_box_type MeshBlock::GetOuterIndexBox() const {
//    auto ibox = GetIndexBox();
//    std::get<0>(ibox) -= GetGhostWidth();
//    std::get<1>(ibox) += GetGhostWidth();
//    return std::move(ibox);
//}
// index_box_type MeshBlock::GetInnerIndexBox() const { return GetIndexBox(); }
//
// index_tuple MeshBlock::GetIndexOrigin() const { return std::get<0>(m_index_box_); }
//
// size_tuple MeshBlock::GetDimensions() const {
//    return std::get<1>(m_index_box_) - std::get<0>(m_index_box_);
//}

}  // namespace engine {
}  // namespace simpla {