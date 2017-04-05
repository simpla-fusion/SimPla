//
// Created by salmon on 17-2-16.
//
#include "Context.h"
#include "Mesh.h"
#include "Task.h"
#include "Worker.h"
#include "simpla/data/all.h"
#include "simpla/geometry/GeoAlgorithm.h"

namespace simpla {
namespace engine {

struct Context::pimpl_s {
    std::shared_ptr<data::DataTable> m_patches_;
    std::map<std::string, std::shared_ptr<Worker>> m_worker_;
    std::map<std::string, std::shared_ptr<Attribute>> m_global_attributes_;
    Atlas m_atlas_;
    Model m_model_;

    bool m_is_initialized_ = false;
};

Context::Context(std::shared_ptr<data::DataTable> const &t) : m_pimpl_(new pimpl_s), concept::Configurable(t) {
    db()->Link("Model", m_pimpl_->m_model_.db());
    db()->Link("Atlas", m_pimpl_->m_atlas_.db());
    m_pimpl_->m_patches_ = std::make_shared<data::DataTable>();
    db()->Link("Patches", *m_pimpl_->m_patches_);
}

Context::~Context() {}
Atlas &Context::GetAtlas() const { return m_pimpl_->m_atlas_; }
Model &Context::GetModel() const { return m_pimpl_->m_model_; }
std::shared_ptr<data::DataTable> Context::GetPatches() const { return m_pimpl_->m_patches_; }

bool Context::SetWorker(std::string const &d_name, std::shared_ptr<Worker> const &p) {
    ASSERT(!IsInitialized());

    auto res = m_pimpl_->m_worker_.emplace(d_name, p);
    if (!res.second) { res.first->second = p; }
    db()->Set("Workers/" + d_name, res.first->second->db());
    return res.second;
}
void Context::RemoveWorker(std::string const &k) {
    ASSERT(!IsInitialized());
    m_pimpl_->m_worker_.erase(k);
}
std::shared_ptr<Worker> Context::GetWorker(std::string const &d_name) const { return m_pimpl_->m_worker_.at(d_name); }

std::map<std::string, std::shared_ptr<Attribute>> const &Context::GetAllAttributes() const {
    return m_pimpl_->m_global_attributes_;
};

bool Context::RegisterAttribute(std::string const &key, std::shared_ptr<Attribute> const &v) {
    ASSERT(!IsInitialized());
    return m_pimpl_->m_global_attributes_.emplace(key, v).second;
}
void Context::DeregisterAttribute(std::string const &key) {
    ASSERT(!IsInitialized());
    m_pimpl_->m_global_attributes_.erase(key);
}
std::shared_ptr<Attribute> const &Context::GetAttribute(std::string const &key) const {
    return m_pimpl_->m_global_attributes_.at(key);
}
void Context::Initialize() {
    ASSERT(!IsInitialized());

    GetModel().Initialize();
    GetAtlas().Initialize();
    db()->Set("Workers/");
    auto domain_p = db()->GetTable("Workers");
    if (domain_p != nullptr) {
        domain_p->Foreach([&](std::string const &s_key, std::shared_ptr<data::DataEntity> const &item) {
            if (!item->isTable()) { return; }
            item->cast_as<DataTable>().SetValue("name",
                                                item->cast_as<DataTable>().GetValue<std::string>("name", s_key));

            auto g_obj_ = GetModel().GetObject(s_key);
            auto view_res = m_pimpl_->m_worker_.emplace(s_key, nullptr);
            if (view_res.first->second == nullptr) {
//                view_res.first->second = std::make_shared<Worker>(std::dynamic_pointer_cast<DataTable>(item), g_obj_);
            } else if (item != nullptr && item->isTable()) {
                view_res.first->second->db()->Set(item->cast_as<data::DataTable>());
            } else {
                WARNING << " ignore data entity [" << s_key << " :" << *item << "]" << std::endl;
            }
            view_res.first->second->Initialize();
            domain_p->Link(s_key, view_res.first->second->db());
            GetModel().AddObject(s_key, view_res.first->second->GetMesh()->GetGeoObject());
        });
    }
    m_pimpl_->m_is_initialized_ = true;
    LOGGER << "Context is initialized!" << std::endl;
}
void Context::Finalize() { m_pimpl_->m_is_initialized_ = false; }
bool Context::IsInitialized() const { return m_pimpl_->m_is_initialized_; }

void Context::Synchronize(int from, int to) {
    if (from >= GetAtlas().GetNumOfLevels() || to >= GetAtlas().GetNumOfLevels()) { return; }
    auto &atlas = GetAtlas();
    for (auto const &src : atlas.GetBlockList(from)) {
        for (auto &dest : atlas.GetBlockList(from)) {
            if (!geometry::CheckOverlap(atlas.GetBlock(src)->GetIndexBox(), atlas.GetBlock(dest)->GetIndexBox())) {
                continue;
            }
            auto s_it = m_pimpl_->m_patches_->Get(std::to_string(src));
            auto d_it = m_pimpl_->m_patches_->Get(std::to_string(dest));
            if (s_it == nullptr || d_it == nullptr || s_it == d_it) { continue; }
            LOGGER << "Synchronize From " << m_pimpl_->m_atlas_.GetBlock(src)->GetIndexBox() << " to "
                   << m_pimpl_->m_atlas_.GetBlock(dest)->GetIndexBox() << " " << std::endl;
            auto &src_data = s_it->cast_as<data::DataTable>();
            src_data.Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &dest_p) {
                auto dest_data = d_it->cast_as<data::DataTable>().Get(key);
                //                        ->cast_as<data::DataTable>();
                if (dest_data == nullptr) { return; }

            });
        }
    }
}
void Context::Advance(Real dt, int level) {
    if (level >= GetAtlas().GetNumOfLevels()) { return; }
    auto &atlas = GetAtlas();
    for (auto const &id : atlas.GetBlockList(level)) {
        auto mblk = m_pimpl_->m_atlas_.GetBlock(id);
        for (auto &v : m_pimpl_->m_worker_) {
            if (!v.second->GetMesh()->GetGeoObject()->CheckOverlap(mblk->GetBoundBox())) { continue; }
            auto res = m_pimpl_->m_patches_->GetTable(std::to_string(id));
            if (res == nullptr) { res = std::make_shared<data::DataTable>(); }
//            v.second->PushData(mblk, res);
            LOGGER << " Worker [ " << std::setw(10) << std::left << v.second->name() << " ] is applied on "
                   << mblk->GetIndexBox() << " id= " << id << std::endl;
            v.second->Advance(0, dt);
//            m_pimpl_->m_patches_->Set(std::to_string(id), v.second->PopData());
        }
    }
    //    m_pimpl_->m_time_ += dt;
};

}  // namespace engine {
}  // namespace simpla {
