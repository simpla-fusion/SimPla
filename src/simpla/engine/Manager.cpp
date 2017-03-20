//
// Created by salmon on 17-2-16.
//
#include "Manager.h"
#include <simpla/data/all.h>
#include <simpla/geometry/GeoAlgorithm.h>
#include "MeshView.h"
#include "Worker.h"
namespace simpla {
namespace engine {

struct Manager::pimpl_s {
    std::map<id_type, std::shared_ptr<data::DataEntity>> m_patches_;
    std::map<std::string, std::shared_ptr<DomainView>> m_views_;
    std::map<std::string, std::shared_ptr<AttributeView>> m_attrs_;
    Atlas m_atlas_;
    Model m_model_;
    Real m_time_ = 0;
};

Manager::Manager() : m_pimpl_(new pimpl_s) {
    db()->Link("Model", m_pimpl_->m_model_.db());
    db()->Link("Atlas", m_pimpl_->m_atlas_.db());
}

Manager::~Manager() {}
Real Manager::GetTime() const { return m_pimpl_->m_time_; }
std::ostream &Manager::Print(std::ostream &os, int indent) const { return db()->Print(os, indent); }

Atlas &Manager::GetAtlas() const { return m_pimpl_->m_atlas_; }

Model &Manager::GetModel() const { return m_pimpl_->m_model_; }

std::shared_ptr<DomainView> Manager::GetDomainView(std::string const &d_name) const {
    return m_pimpl_->m_views_.at(d_name);
}

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
            auto s_it = m_pimpl_->m_patches_.find(src);
            auto d_it = m_pimpl_->m_patches_.find(dest);
            if (s_it == m_pimpl_->m_patches_.end() || d_it == m_pimpl_->m_patches_.end() || s_it == d_it) { continue; }
            LOGGER << "Synchronize From " << m_pimpl_->m_atlas_.GetBlock(s_it->first)->GetIndexBox() << " to "
                   << m_pimpl_->m_atlas_.GetBlock(d_it->first)->GetIndexBox() << " " << std::endl;
            auto &src_data = s_it->second->cast_as<data::DataTable>();
            src_data.Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &dest_p) {
                auto dest_data = d_it->second->cast_as<data::DataTable>().Get(key);
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
            auto res = m_pimpl_->m_patches_.emplace(id, nullptr);
            if (res.first->second == nullptr) { res.first->second = std::make_shared<data::DataTable>(); }
            v.second->PushData(mblk, res.first->second);
            LOGGER << " DomainView [ " << std::setw(10) << std::left << v.second->name() << " ] is applied on "
                   << mblk->GetIndexBox() << std::endl;
            v.second->Run(dt);
            std::tie(std::ignore, res.first->second) = v.second->PopData();
        }
    }
    m_pimpl_->m_time_ += dt;
};

void Manager::Initialize() {
    LOGGER << "Manager " << name() << " is initializing!" << std::endl;
    GetModel().Initialize();
    GetAtlas().Initialize();
    db()->Set("DomainView/");
    auto &domain_t = *db()->GetTable("DomainView");
    domain_t.Foreach([&](std::string const &s_key, std::shared_ptr<data::DataEntity> const &item) {
        item->cast_as<DataTable>().SetValue("name", item->cast_as<DataTable>().GetValue<std::string>("name", s_key));

        auto g_obj_ = GetModel().GetObject(s_key);
        auto view_res = m_pimpl_->m_views_.emplace(s_key, nullptr);
        if (view_res.first->second == nullptr) {
            view_res.first->second = std::make_shared<DomainView>(item, g_obj_);
        } else if (item != nullptr && item->isTable()) {
            view_res.first->second->db()->Set(*std::dynamic_pointer_cast<data::DataTable>(item));
        }
        // else { WARNING << " ignore data entity :" << *item << std::endl;}

        domain_t.Set(s_key, view_res.first->second->db(), true);
        view_res.first->second->Initialize(domain_t.Get(s_key), g_obj_);
        GetModel().AddObject(s_key, view_res.first->second->GetMesh()->GetGeoObject());
    });
    SPObject::Tag();
    LOGGER << "Manager [" << name() << "] is initialized!" << std::endl;
}
bool Manager::Update() { return SPObject::Update(); };
}  // namespace engine {
}  // namespace simpla {
