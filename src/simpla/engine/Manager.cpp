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
    std::map<id_type, std::shared_ptr<Patch>> m_patches_;
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
            LOGGER << "Synchronize From " << s_it->second->GetMeshBlock()->GetIndexBox() << " to "
                   << s_it->second->GetMeshBlock()->GetIndexBox() << " " << std::endl;
            auto &src_data = s_it->second->GetAllDataBlock();
            for (auto const &item : src_data) {
                auto dest_data = d_it->second->GetDataBlock(item.first);
                if (dest_data == nullptr) { continue; }
            }
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
            if (res.first->second == nullptr) { res.first->second = std::make_shared<Patch>(mblk); }
            v.second->Dispatch(res.first->second);
            LOGGER << " Run " << v.second->name() << " at " << res.first->second->GetMeshBlock()->GetIndexBox()
                   << std::endl;
            v.second->Run(dt);
        }
    }
    m_pimpl_->m_time_ += dt;
};

bool Manager::Update() { return SPObject::Update(); };

void Manager::Initialize() {
    LOGGER << "Manager " << name() << " is initializing!" << std::endl;
    GetModel().Initialize();
    GetAtlas().Initialize();
    auto domain_view_list = db()->Get("DomainView");
    if (domain_view_list == nullptr || !domain_view_list->isTable()) { return; }
    auto &domain_t = domain_view_list->cast_as<data::DataTable>();
    domain_t.Foreach([&](std::string const &s_key, std::shared_ptr<data::DataEntity> const &item) {
        auto res = m_pimpl_->m_views_.emplace(s_key, nullptr);
        if (res.first->second == nullptr) {
            res.first->second = std::make_shared<DomainView>(
                (item != nullptr && item->isTable()) ? std::dynamic_pointer_cast<data::DataTable>(item) : nullptr);
        } else {
            if (item != nullptr && item->isTable()) {
                res.first->second->db()->Set(*std::dynamic_pointer_cast<data::DataTable>(item));
            }
        }
        domain_t.Set(s_key, res.first->second->db(), true);
        res.first->second->name(s_key);
        res.first->second->Initialize();
    });
    SPObject::Tag();
    LOGGER << "Manager " << name() << " is initialized!" << std::endl;
}
}  // namespace engine {
}  // namespace simpla {
