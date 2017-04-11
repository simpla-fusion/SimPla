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
    std::map<id_type, std::shared_ptr<Patch>> m_patches_;
    std::map<std::string, std::shared_ptr<Worker>> m_workers_;
    std::map<std::string, std::shared_ptr<Attribute>> m_global_attributes_;
    Atlas m_atlas_;
    Model m_model_;
    bool m_is_initialized_ = false;
};

Context::Context() : m_pimpl_(new pimpl_s) {}
Context::~Context() {}
// Atlas &Context::GetAtlas() const { return m_pimpl_->m_atlas_; }
// Model &Context::GetModel() const { return m_pimpl_->m_model_; }
// std::map<id_type, std::shared_ptr<Patch>> const &Context::GetPatches() const { return m_pimpl_->m_patches_; }
//
// bool Context::RegisterWorker(std::string const &d_name, std::shared_ptr<Worker> const &p) {
//    ASSERT(!IsInitialized());
//
//    auto res = m_pimpl_->m_workers_.emplace(d_name, p);
//    if (!res.second) { res.first->second = p; }
//    db()->Set("Workers/" + d_name, res.first->second->db());
//    return res.second;
//}
// void Context::DeregisterWorker(std::string const &k) {
//    ASSERT(!IsInitialized());
//    m_pimpl_->m_workers_.erase(k);
//}
// std::shared_ptr<Worker> Context::GetWorker(std::string const &d_name) const { return m_pimpl_->m_workers_.at(d_name);
// }
//
// std::map<std::string, std::shared_ptr<Attribute>> const &Context::GetAllAttributes() const {
//    return m_pimpl_->m_global_attributes_;
//};
//
// bool Context::RegisterAttribute(std::string const &key, std::shared_ptr<Attribute> const &v) {
//    ASSERT(!IsInitialized());
//    return m_pimpl_->m_global_attributes_.emplace(key, v).second;
//}
// void Context::DeregisterAttribute(std::string const &key) {
//    ASSERT(!IsInitialized());
//    m_pimpl_->m_global_attributes_.erase(key);
//}
// std::shared_ptr<Attribute> const &Context::GetAttribute(std::string const &key) const {
//    return m_pimpl_->m_global_attributes_.at(key);
//}
void Context::Initialize() {
    ASSERT(!IsInitialized());

    //    GetModel().Initialize();
    //    GetAtlas().Initialize();
    //    auto workers_t = db()->GetTable("Workers");
    //    GetModel().GetMaterial().Foreach([]() {
    //
    //    });
    //
    //    for (auto const &item : GetModel().GetAllMaterial()) {}
    //
    //    db()->GetTable("Domains")->Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &v) {
    //        if (!v->isTable()) { return; }
    //        auto const &t = v->cast_as<data::DataTable>();
    //
    //        std::shared_ptr m(GLOBAL_MESHVIEW_FACTORY.Create(t.GetTable("Mesh"),
    //                                                         GetModel().AddObject(key,
    //                                                         t.GetTable("Geometry")).first));
    //
    //        m_pimpl_->m_worker_.emplace(key, std::make_shared<Worker>(t.GetTable("Worker"), m));
    //
    //    });
    //    for (auto const &item : GetModel().GetAll()) {
    //        auto worker_res = m_pimpl_->m_worker_.emplace(item.first, nullptr);
    //        if (worker_res.first->second == nullptr) {
    //            worker_res.first->second = std::make_shared<Worker>(workers_t->GetTable(item.first), nullptr,
    //            item.second);
    //        }
    //        workers_t->Link(item.first, worker_res.first->second->db());
    //        // TODO register attributes
    //    }

    //    std::shared_ptr<geometry::GeoObject> geo = g;
    //    if (geo == nullptr) { geo.reset(GLOBAL_GEO_OBJECT_FACTORY.Create(db()->GetTable("Geometry"))); }
    //    m_pimpl_->m_chart_.reset(GLOBAL_MESHVIEW_FACTORY.Create(db()->GetTable("Mesh"), geo));
    //
    //    m_pimpl_->m_is_initialized_ = true;
    //    LOGGER << "Context is initialized!" << std::endl;
}
void Context::Finalize() {
    m_pimpl_->m_is_initialized_ = false;
    for (auto const &item : m_pimpl_->m_workers_) { item.second->Finalize(); }
    //    GetModel().Finalize();
    //    GetAtlas().Finalize();
}
bool Context::IsInitialized() const { return m_pimpl_->m_is_initialized_; }

void Context::Synchronize(int from, int to) {
    //    if (from >= GetAtlas().GetNumOfLevels() || to >= GetAtlas().GetNumOfLevels()) { return; }
    //    auto &atlas = GetAtlas();
    //    for (auto const &src : atlas.Level(from)) {
    //        for (auto const &dest : atlas.Level(from)) {
    //            if (!geometry::CheckOverlap(src->GetIndexBox(), dest->GetIndexBox())) { continue; }
    //            auto s_it = m_pimpl_->m_patches_.find(src->GetGUID());
    //            auto d_it = m_pimpl_->m_patches_.find(dest->GetGUID());
    //            if (s_it == m_pimpl_->m_patches_.end() || d_it == m_pimpl_->m_patches_.end() || s_it == d_it) {
    //            continue; }
    //            //            LOGGER << "Synchronize From " << m_pimpl_->m_atlas_.GetBlock(src)->GetIndexBox() << " to
    //            "
    //            //                   << m_pimpl_->m_atlas_.GetBlock(dest)->GetIndexBox() << " " << std::endl;
    //            //            auto &src_data = s_it->cast_as<data::DataTable>();
    //            //            src_data.Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const
    //            &dest_p)
    //            //            {
    //            //                auto dest_data = d_it->cast_as<data::DataTable>().Get(key);
    //            //                //                        ->cast_as<data::DataTable>();
    //            //                if (dest_data == nullptr) { return; }
    //            //
    //            //            });
    //        }
    //    }
}
void Context::Advance(Real time_now, Real dt, int level){
    //    if (level >= GetAtlas().GetNumOfLevels()) { return; }
    //
    //    for (auto const &g_item : GetModel().GetAll()) {
    //        auto w = m_pimpl_->m_workers_.find(g_item.first);
    //        if (w == m_pimpl_->m_workers_.end()) { continue; }
    //        for (auto const &mblk : GetAtlas().Level(level)) {
    //            if (!g_item.second->CheckOverlap(mblk->GetBoundBox())) { continue; }
    //
    //            auto p = m_pimpl_->m_patches_[mblk->GetGUID()];
    //            if (p == nullptr) {
    //                p = std::make_shared<Patch>();
    //                //                p->PushMeshBlock(mblk);
    //            }
    //            w->second->Push(p);
    //            LOGGER << " Worker [ " << std::setw(10) << std::left << w->second->name() << " ] is applied on "
    //                   << mblk->GetIndexBox() << " GeoObject id= " << g_item.first << std::endl;
    //            w->second->Advance(time_now, dt);
    //            m_pimpl_->m_patches_[mblk->GetGUID()] = w->second->Pop();
    //        }
    //    }
};

}  // namespace engine {
}  // namespace simpla {
