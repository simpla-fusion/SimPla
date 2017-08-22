//
// Created by salmon on 17-4-5.
//
#include "simpla/SIMPLA_config.h"

#include "Schedule.h"

#include <map>
#include <string>

#include "Atlas.h"
#include "Attribute.h"
#include "Context.h"
#include "Mesh.h"

#include "simpla/data/DataNode.h"

namespace simpla {
namespace engine {
struct Schedule::pimpl_s {
    size_type m_step_ = 0;
    size_type m_max_step_ = 0;
    size_type m_check_point_interval_ = 1;
    size_type m_dump_interval_ = 0;

    std::shared_ptr<Context> m_ctx_ = nullptr;
    std::shared_ptr<Atlas> m_atlas_ = nullptr;
};

Schedule::Schedule() : m_pimpl_(new pimpl_s){};
Schedule::~Schedule() { delete m_pimpl_; };

void Schedule::Serialize(const std::shared_ptr<data::DataNode> &cfg) const {
    base_type::Serialize(cfg);
    db()->Set(cfg);
    auto tdb = std::dynamic_pointer_cast<data::DataTable>(cfg);
    if (tdb != nullptr) { tdb->SetValue("CheckPointInterval", GetCheckPointInterval()); }
}

void Schedule::Deserialize(const std::shared_ptr<const data::DataNode> &cfg) {
    base_type::Deserialize(cfg);
    m_pimpl_->m_ctx_ = Atlas::New(cfg->NewNode("Atlas"));
    SetCheckPointInterval(static_cast<size_type>(db().GetValue<int>("CheckPointInterval", 1)));
    //    m_data_io_ = std::make_shared<data::DataIOPort>(cfg->Get<std::string>("DataIOPort", ""));
}

size_type Schedule::GetNumberOfStep() const { return m_pimpl_->m_step_; }
void Schedule::SetMaxStep(size_type s) { m_pimpl_->m_max_step_ = s; }
size_type Schedule::GetMaxStep() const { return m_pimpl_->m_max_step_; }


void Schedule::NextStep() { ++m_pimpl_->m_step_; }

bool Schedule::Done() const { return m_pimpl_->m_max_step_ == 0 ? false : m_pimpl_->m_step_ >= m_pimpl_->m_max_step_; }

void Schedule::CheckPoint() const {
    //    data::DataTable t_cfg;
    //    GetAtlas()->Serialize(t_cfg);
    //    t_cfg->Set("Step", m_pimpl_->m_step_);
    //    m_ctx_->GetMesh()->Serialize(t_cfg->GetTable("Mesh"));
    //    m_data_io_->Set(t);
    //    m_data_io_->Flush();
}

void Schedule::Dump() const { UNIMPLEMENTED; }

void Schedule::Run() {
    while (!Done()) {
        VERBOSE << " [ STEP:" << std::setw(5) << m_pimpl_->m_step_ << " START ] " << std::endl;
        if (m_pimpl_->m_step_ == 0) { CheckPoint(); }
        Synchronize();
        NextStep();
        if (m_pimpl_->m_check_point_interval_ > 0 && m_pimpl_->m_step_ % m_pimpl_->m_check_point_interval_ == 0) {
            CheckPoint();
        };
        if (m_pimpl_->m_dump_interval_ > 0 && m_pimpl_->m_step_ % m_pimpl_->m_dump_interval_ == 0) { Dump(); };

        VERBOSE << " [ STEP:" << std::setw(5) << m_pimpl_->m_step_ - 1 << " STOP  ] " << std::endl;
    }
}

void Schedule::DoInitialize() { SPObject::DoInitialize(); }
void Schedule::DoFinalize() { SPObject::DoFinalize(); }
void Schedule::DoUpdate() { SPObject::DoUpdate(); }
void Schedule::DoTearDown() { SPObject::DoTearDown(); }
void Schedule::Synchronize() {
    //    auto &atlas = GetContext()->GetAtlas();
    //    if (from_level >= atlas.GetNumOfLevel() || to_level >= atlas.GetNumOfLevel()) { return; }
    //    for (auto const &src : atlas.Level(from_level)) {
    //        for (auto const &dest : atlas.Level(from_level)) {
    //            if (!geometry::CheckOverlap(src->IndexBox(), dest->IndexBox())) { continue; }
    //            //            auto s_it = m_pack_->m_patches_.find(src->GetID());
    //            //            auto d_it = m_pack_->m_patches_.find(dest->GetID());
    //            //            if (s_it == m_pack_->m_patches_.end() || d_it == m_pack_->m_patches_.end() || s_it ==
    //            d_it)
    //            //            { continue; }
    //            //            LOGGER << "Synchronize From " << m_pack_->m_atlas_.GetMeshBlock(src)->IndexBox() <<
    //            " to
    //            "
    //            //                   << m_pack_->m_atlas_.GetMeshBlock(dest)->IndexBox() << " " << std::endl;
    //            //            auto &src_data = s_it->cast_as<data::DataTable>();
    //            //            src_data.Foreach([&](std::string const &key, std::shared_ptr<data::DataNode> const
    //            &dest_p)
    //            //            {
    //            //                auto dest_data = d_it->cast_as<data::DataTable>().Serialize(key);
    //            //                if (dest_data == nullptr) { return; }
    //            //            });
    //        }
    //    }
}

}  // namespace engine{
}  // namespace simpla{