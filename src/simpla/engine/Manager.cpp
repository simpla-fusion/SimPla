//
// Created by salmon on 17-2-16.
//
#include "Manager.h"
#include <simpla/data/DataUtility.h>
#include <simpla/data/all.h>
#include <simpla/geometry/GeoAlgorithm.h>

#include "MeshView.h"
#include "Worker.h"
namespace simpla {
namespace engine {

struct Manager::pimpl_s {
    std::shared_ptr<data::DataTable> m_patches_;
    std::map<std::string, std::shared_ptr<DomainView>> m_views_;
    std::map<std::string, std::shared_ptr<AttributeView>> m_attrs_;
    Atlas m_atlas_;
    Model m_model_;
    Real m_time_ = 0;
    std::shared_ptr<data::DataTable> m_db_;
};

Manager::Manager(std::shared_ptr<data::DataEntity> const &t) : m_pimpl_(new pimpl_s), concept::Configurable(t) {
    db()->Link("Model", m_pimpl_->m_model_.db());
    db()->Link("Atlas", m_pimpl_->m_atlas_.db());
    m_pimpl_->m_patches_ = std::make_shared<data::DataTable>();
    db()->Link("Patches", *m_pimpl_->m_patches_);
}

Manager::~Manager() {}
Real Manager::GetTime() const { return m_pimpl_->m_time_; }
Atlas &Manager::GetAtlas() const { return m_pimpl_->m_atlas_; }
Model &Manager::GetModel() const { return m_pimpl_->m_model_; }
std::shared_ptr<data::DataTable> Manager::GetPatches() const { return m_pimpl_->m_patches_; }
std::shared_ptr<DomainView> Manager::GetDomainView(std::string const &d_name) const {
    return m_pimpl_->m_views_.at(d_name);
}
std::map<std::string, std::shared_ptr<DomainView>> const &Manager::GetAllDomainViews() const {
    return m_pimpl_->m_views_;
};

void Manager::SetDomainView(std::string const &d_name, std::shared_ptr<data::DataTable> const &p) {
    db()->Set("DomainView/" + d_name, *p, false);
}

void Manager::Synchronize(int from, int to) {
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
void Manager::Advance(Real dt, int level) {
    if (level >= GetAtlas().GetNumOfLevels()) { return; }
    auto &atlas = GetAtlas();
    for (auto const &id : atlas.GetBlockList(level)) {
        auto mblk = m_pimpl_->m_atlas_.GetBlock(id);
        for (auto &v : m_pimpl_->m_views_) {
            if (!v.second->GetMesh()->GetGeoObject()->CheckOverlap(mblk->GetBoundBox())) { continue; }
            auto res = m_pimpl_->m_patches_->Get(std::to_string(id));
            if (res == nullptr) { res = std::make_shared<data::DataTable>(); }
            v.second->PushData(mblk, res);
            LOGGER << " DomainView [ " << std::setw(10) << std::left << v.second->name() << " ] is applied on "
                   << mblk->GetIndexBox() << " id= " << id << std::endl;
            v.second->Run(dt);
            auto t = v.second->PopData().second;
            m_pimpl_->m_patches_->Set(std::to_string(id), t);
        }
    }
    m_pimpl_->m_time_ += dt;
};

void Manager::Initialize() {
    GetModel().Initialize();
    GetAtlas().Initialize();
    db()->Set("DomainView/");
    auto &domain_t = *db()->GetTable("DomainView");
    domain_t.Foreach([&](std::string const &s_key, std::shared_ptr<data::DataEntity> const &item) {
        if (!item->isTable()) { return; }
        item->cast_as<DataTable>().SetValue("name", item->cast_as<DataTable>().GetValue<std::string>("name", s_key));

        auto g_obj_ = GetModel().GetObject(s_key);
        auto view_res = m_pimpl_->m_views_.emplace(s_key, nullptr);
        if (view_res.first->second == nullptr) {
            view_res.first->second = std::make_shared<DomainView>(item, g_obj_);
        } else if (item != nullptr && item->isTable()) {
            view_res.first->second->db()->Set(item->cast_as<data::DataTable>());
        } else {
            WARNING << " ignore data entity [" << s_key << " :" << *item << "]" << std::endl;
        }
        view_res.first->second->Initialize();
        domain_t.Link(s_key, view_res.first->second->db());

        GetModel().AddObject(s_key, view_res.first->second->GetGeoObject());
    });

    LOGGER << "Manager is initialized!" << std::endl;
}
}  // namespace engine {
}  // namespace simpla {
