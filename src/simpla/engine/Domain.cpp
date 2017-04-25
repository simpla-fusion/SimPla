//
// Created by salmon on 17-4-1.
//
#include "Domain.h"
#include <simpla/SIMPLA_config.h>
#include <set>
#include "Attribute.h"
#include "MeshBase.h"
#include "MeshBlock.h"
#include "Patch.h"
#include "SPObject.h"
#include "Task.h"
#include "Worker.h"
namespace simpla {
namespace engine {

struct Domain::pimpl_s {
    std::shared_ptr<Chart> m_chart_ = nullptr;
    std::shared_ptr<MeshBase> m_mesh_ = nullptr;
    std::shared_ptr<Worker> m_worker_ = nullptr;
    std::shared_ptr<geometry::GeoObject> m_geo_obj_ = nullptr;
    std::map<std::shared_ptr<geometry::GeoObject>, std::shared_ptr<Worker>> m_boundary_;
};
Domain::Domain() : SPObject(), m_pimpl_(new pimpl_s) {}
Domain::~Domain() { Finalize(); }

void Domain::Initialize() {}
void Domain::SetUp() {
    m_pimpl_->m_worker_->GetMesh()->SetChart(m_pimpl_->m_chart_);
    m_pimpl_->m_worker_->SetUp();
}
void Domain::TearDown() {}
void Domain::Finalize() {}
void Domain::InitializeData(Patch *p, Real time_now) {
    m_pimpl_->m_worker_->Push(p);
    m_pimpl_->m_worker_->InitializeCondition(time_now);
    m_pimpl_->m_worker_->Pop(p);
}

std::shared_ptr<data::DataTable> Domain::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    res->SetValue("Type", "Domain");
    if (m_pimpl_->m_chart_ != nullptr) { res->Link("Chart", m_pimpl_->m_chart_->Serialize()); }
    if (m_pimpl_->m_geo_obj_ != nullptr) { res->Link("GeoObject", m_pimpl_->m_geo_obj_->Serialize()); }
    if (m_pimpl_->m_worker_ != nullptr) { res->Link("Worker", m_pimpl_->m_worker_->Serialize()); }
    return res;
}
void Domain::Deserialize(std::shared_ptr<data::DataTable> t) {
    m_pimpl_->m_geo_obj_ = geometry::GeoObject::Create(t->GetTable("GeoObject"));
    SetChart(t->GetValue<std::string>("Chart/Type"));
    SetMesh(t->GetValue<std::string>("Topology/Type"));
    SetWorker(t->GetValue<std::string>("Worker/Type"));
    // TODO: unfinished
}
void Domain::Register(AttributeGroup *attr_grp) { GetMesh()->Register(attr_grp); }
void Domain::Deregister(AttributeGroup *attr_grp) { GetMesh()->Deregister(attr_grp); }

void Domain::SetGeoObject(std::shared_ptr<geometry::GeoObject> g) { m_pimpl_->m_geo_obj_ = g; }
std::shared_ptr<geometry::GeoObject> Domain::GetGeoObject() const { return m_pimpl_->m_geo_obj_; }

std::shared_ptr<Chart> Domain::CreateChart(std::string const &chart_s) const { return Chart::Create(chart_s); }
void Domain::SetChart(std::string const &s) { SetChart(CreateChart(s)); };
void Domain::SetChart(std::shared_ptr<Chart> m) { m_pimpl_->m_chart_ = m; };
std::shared_ptr<Chart> Domain::GetChart() const { return m_pimpl_->m_chart_; }

std::shared_ptr<MeshBase> Domain::CreateMesh(std::string const &s) const {
    auto res = MeshBase::Create("Mesh<" + GetChart()->GetClassName() + "," + s + ">", GetChart());
    res->SetUp();
    return res;
}
void Domain::SetMesh(std::string const &s) { SetMesh(CreateMesh(s)); }
void Domain::SetMesh(std::shared_ptr<MeshBase> m) { m_pimpl_->m_mesh_ = m; }
std::shared_ptr<MeshBase> Domain::GetMesh() const { return m_pimpl_->m_mesh_; }

std::shared_ptr<Worker> Domain::CreateWorker(std::string const &worker_s) const {
    auto res = Worker::Create(worker_s + "<" + GetMesh()->GetClassName() + ">", GetMesh());
    res->SetUp();
    return res;
}
void Domain::SetWorker(std::string const &worker_s) { SetWorker(CreateWorker(worker_s)); }
void Domain::SetWorker(std::shared_ptr<Worker> w) {
    m_pimpl_->m_worker_ = w;
    if (w->GetMesh()->GetChart() != nullptr) { SetChart(w->GetMesh()->GetChart()); }
}
std::shared_ptr<Worker> Domain::GetWorker() const { return m_pimpl_->m_worker_; }

void Domain::AddBoundaryCondition(std::string const &worker_s, std::shared_ptr<geometry::GeoObject> g) {
    auto res = Worker::Create(worker_s + "<BoundaryMeshBase>", std::make_shared<BoundaryMeshBase>(GetMesh().get()));
    res->SetUp();
    AddBoundaryCondition(res, g);
}
void Domain::AddBoundaryCondition(std::shared_ptr<Worker> w, std::shared_ptr<geometry::GeoObject> g) {
    m_pimpl_->m_boundary_.emplace(g, w);
}

void Domain::InitializeCondition(Patch *p, Real time_now) {
    ASSERT(m_pimpl_->m_worker_ != nullptr);
    m_pimpl_->m_worker_->Push(p);
    m_pimpl_->m_worker_->InitializeCondition(time_now);
    m_pimpl_->m_worker_->Pop(p);
}

void Domain::BoundaryCondition(Patch *p, Real time_now, Real time_dt) {
    ASSERT(m_pimpl_->m_worker_ != nullptr);
    m_pimpl_->m_worker_->Push(p);
    m_pimpl_->m_worker_->BoundaryCondition(time_now, time_dt);
    m_pimpl_->m_worker_->Pop(p);
}

void Domain::Advance(Patch *p, Real time_now, Real time_dt) {
    ASSERT(m_pimpl_->m_worker_ != nullptr);
    m_pimpl_->m_worker_->Push(p);
    m_pimpl_->m_worker_->Advance(time_now, time_dt);
    m_pimpl_->m_worker_->Pop(p);
}

// void Model::UpdatePatch(Real data_time, Real dt) {
//    PreProcess();
//    //
//    //    index_type const* lower = m_tags_.lower();
//    //    index_type const* upper = m_tags_.upper();
//    //
//    //    index_type ib = lower[0];
//    //    index_type ie = upper[0];
//    //    index_type jb = lower[1];
//    //    index_type je = upper[1];
//    //    index_type kb = lower[2];
//    //    index_type ke = upper[2];
//    //
//    //    for (index_type i = ib; i < ie; ++i)
//    //        for (index_type j = jb; j < je; ++j)
//    //            for (index_type k = kb; k < ke; ++k) {
//    //                auto x = m_chart_->mesh_block()->point(i, j, k);
//    //                auto& GetTag = m_tags_(i, j, k, 0);
//    //
//    //                GetTag = VACUUM;
//    //
//    //                for (auto const& obj : m_g_objs_) {
//    //                    if (obj.second->check_inside(x)) { GetTag |= obj.first; }
//    //                }
//    //            }
//    //    for (index_type i = ib; i < ie - 1; ++i)
//    //        for (index_type j = jb; j < je - 1; ++j)
//    //            for (index_type k = kb; k < ke - 1; ++k) {
//    //                m_tags_(i, j, k, 1) = m_tags_(i, j, k, 0) | m_tags_(i + 1, j, k, 0);
//    //                m_tags_(i, j, k, 2) = m_tags_(i, j, k, 0) | m_tags_(i, j + 1, k, 0);
//    //                m_tags_(i, j, k, 4) = m_tags_(i, j, k, 0) | m_tags_(i, j, k + 1, 0);
//    //            }
//    //
//    //    for (index_type i = ib; i < ie - 1; ++i)
//    //        for (index_type j = jb; j < je - 1; ++j)
//    //            for (index_type k = kb; k < ke - 1; ++k) {
//    //                m_tags_(i, j, k, 3) = m_tags_(i, j, k, 1) | m_tags_(i, j + 1, k, 1);
//    //                m_tags_(i, j, k, 5) = m_tags_(i, j, k, 1) | m_tags_(i, j, k + 1, 1);
//    //                m_tags_(i, j, k, 6) = m_tags_(i, j + 1, k, 1) | m_tags_(i, j, k + 1, 1);
//    //            }
//    //
//    //    for (index_type i = ib; i < ie - 1; ++i)
//    //        for (index_type j = jb; j < je - 1; ++j)
//    //            for (index_type k = kb; k < ke - 1; ++k) {
//    //                m_tags_(i, j, k, 7) = m_tags_(i, j, k, 3) | m_tags_(i, j, k + 1, 3);
//    //            }
//};
//
// void Model::Finalize(Real data_time, Real dt) {
//    m_range_cache_.Disconnect(m_chart_->mesh_block()->id());
//    m_interface_cache_.Disconnect(m_chart_->mesh_block()->id());
//    PostProcess();
//};
//
// Range<id_type> const& Model::select(int GetIFORM, std::string const& GetTag) { return select(GetIFORM,
// m_g_name_map_.at(GetTag));
// }
//
// Range<id_type> const& Model::select(int GetIFORM, int GetTag) {
//    //    typedef EntityIdCoder M;
//    //
//    //    try {
//    //        return m_range_cache_.at(GetIFORM).at(GetTag);
//    //    } catch (...) {}
//    //
//    //    const_cast<this_type*>(this)->m_range_cache_[GetIFORM].emplace(
//    //        std::make_pair(GetTag, Range<id_type>(std::make_shared<UnorderedRange<id_type>>())));
//    //
//    //    auto& res = *m_range_cache_.at(GetIFORM).at(GetTag).self().template as<UnorderedRange<id_type>>();
//    //
//    //    index_type const* lower = m_tags_.lower();
//    //    index_type const* upper = m_tags_.upper();
//    //
//    //    index_type ib = lower[0];
//    //    index_type ie = upper[0];
//    //    index_type jb = lower[1];
//    //    index_type je = upper[1];
//    //    index_type kb = lower[2];
//    //    index_type ke = upper[2];
//    //
//    //#define _CAS(I, J, K, L) \
////    if (I >= 0 && J >= 0 && K >= 0 && ((m_tags_(I, J, K, L) & GetTag) == GetTag)) { res.Connect(M::pack_index(I, J, K, L)); }
//    //
//    //    switch (GetIFORM) {
//    //        case VERTEX:
//    //#pragma omp parallel for
//    //            for (index_type i = ib; i < ie; ++i)
//    //                for (index_type j = jb; j < je; ++j)
//    //                    for (index_type k = kb; k < ke; ++k) { _CAS(i, j, k, 0); }
//    //
//    //            break;
//    //        case EDGE:
//    //#pragma omp parallel for
//    //            for (index_type i = ib; i < ie - 1; ++i)
//    //                for (index_type j = jb; j < je - 1; ++j)
//    //                    for (index_type k = kb; k < ke - 1; ++k) {
//    //                        _CAS(i, j, k, 1);
//    //                        _CAS(i, j, k, 2);
//    //                        _CAS(i, j, k, 4);
//    //                    }
//    //            break;
//    //        case FACE:
//    //#pragma omp parallel for
//    //            for (index_type i = ib; i < ie - 1; ++i)
//    //                for (index_type j = jb; j < je - 1; ++j)
//    //                    for (index_type k = kb; k < ke - 1; ++k) {
//    //                        _CAS(i, j, k, 3);
//    //                        _CAS(i, j, k, 5);
//    //                        _CAS(i, j, k, 6);
//    //                    }
//    //            break;
//    //        case VOLUME:
//    //#pragma omp parallel for
//    //            for (index_type i = ib; i < ie - 1; ++i)
//    //                for (index_type j = jb; j < je - 1; ++j)
//    //                    for (index_type k = kb; k < ke - 1; ++k) { _CAS(i, j, k, 7); }
//    //            break;
//    //        default:
//    //            break;
//    //    }
//    //#undef _CAS
//    //    return m_range_cache_.at(GetIFORM).at(GetTag);
//    //    ;
//}
//
///**
// *  id < 0 out of surface
// *       = 0 on surface
// *       > 0 in surface
// */
// Range<id_type> const& Model::interface(int GetIFORM, const std::string& s_in, const std::string& s_out) {
//    return interface(GetIFORM, m_g_name_map_.at(s_in), m_g_name_map_.at(s_out));
//}
//
// Range<id_type> const& Model::interface(int GetIFORM, int tag_in, int tag_out) {
//    //    try {
//    //        return m_interface_cache_.at(GetIFORM).at(tag_in).at(tag_out);
//    //    } catch (...) {}
//    //
//    //    typedef EntityIdCoder M;
//    //
//    //    const_cast<this_type*>(this)->m_interface_cache_[GetIFORM][tag_in].emplace(
//    //        std::make_pair(tag_out, Range<id_type>(std::make_shared<UnorderedRange<id_type>>())));
//    //
//    //    auto& res = *const_cast<this_type*>(this)
//    //                     ->m_interface_cache_.at(GetIFORM)
//    //                     .at(tag_in)
//    //                     .at(tag_out)
//    //                     .self()
//    //                     .template as<UnorderedRange<id_type>>();
//    //
//    //    index_type const* lower = m_tags_.lower();
//    //    index_type const* upper = m_tags_.upper();
//    //
//    //    index_type ib = lower[0];
//    //    index_type ie = upper[0];
//    //    index_type jb = lower[1];
//    //    index_type je = upper[1];
//    //    index_type kb = lower[2];
//    //    index_type ke = upper[2];
//    //
//    //    int v_tag = tag_in | tag_out;
//    //#pragma omp parallel for
//    //    for (index_type i = ib; i < ie - 1; ++i)
//    //        for (index_type j = jb; j < je - 1; ++j)
//    //            for (index_type k = kb; k < ke - 1; ++k) {
//    //                if ((m_tags_(i, j, k, 7) & v_tag) != v_tag) { continue; }
//    //#define _CAS(I, J, K, L) \
////    if (I >= 0 && J >= 0 && K >= 0 && m_tags_(I, J, K, L) == tag_in) { res.Connect(M::pack_index(I, J, K, L)); }
//    //                switch (GetIFORM) {
//    //                    case VERTEX:
//    //                        _CAS(i + 0, j + 0, k + 0, 0);
//    //                        _CAS(i + 1, j + 0, k + 0, 0);
//    //                        _CAS(i + 0, j + 1, k + 0, 0);
//    //                        _CAS(i + 1, j + 1, k + 0, 0);
//    //                        _CAS(i + 0, j + 0, k + 1, 0);
//    //                        _CAS(i + 1, j + 0, k + 1, 0);
//    //                        _CAS(i + 0, j + 1, k + 1, 0);
//    //                        _CAS(i + 1, j + 1, k + 1, 0);
//    //
//    //                        break;
//    //                    case EDGE:
//    //                        _CAS(i + 0, j + 0, k + 0, 1);
//    //                        _CAS(i + 0, j + 1, k + 0, 1);
//    //                        _CAS(i + 0, j + 0, k + 1, 1);
//    //                        _CAS(i + 0, j + 1, k + 1, 1);
//    //
//    //                        _CAS(i + 0, j + 0, k + 0, 2);
//    //                        _CAS(i + 1, j + 0, k + 0, 2);
//    //                        _CAS(i + 0, j + 0, k + 1, 2);
//    //                        _CAS(i + 1, j + 0, k + 1, 2);
//    //
//    //                        _CAS(i + 0, j + 0, k + 0, 4);
//    //                        _CAS(i + 0, j + 1, k + 0, 4);
//    //                        _CAS(i + 1, j + 0, k + 0, 4);
//    //                        _CAS(i + 1, j + 1, k + 0, 4);
//    //                        break;
//    //                    case FACE:
//    //                        _CAS(i + 0, j + 0, k + 0, 3);
//    //                        _CAS(i + 0, j + 0, k + 1, 3);
//    //
//    //                        _CAS(i + 0, j + 0, k + 0, 5);
//    //                        _CAS(i + 0, j + 1, k + 0, 5);
//    //
//    //                        _CAS(i + 0, j + 0, k + 0, 6);
//    //                        _CAS(i + 0, j + 0, k + 1, 6);
//    //                        break;
//    //                    case VOLUME:
//    //                        _CAS(i - 1, j - 1, k - 1, 7);
//    //                        _CAS(i + 0, j - 1, k - 1, 7);
//    //                        _CAS(i + 1, j - 1, k - 1, 7);
//    //                        _CAS(i - 1, j + 0, k - 1, 7);
//    //                        _CAS(i + 0, j + 0, k - 1, 7);
//    //                        _CAS(i + 1, j + 0, k - 1, 7);
//    //                        _CAS(i - 1, j + 1, k - 1, 7);
//    //                        _CAS(i + 0, j + 1, k - 1, 7);
//    //                        _CAS(i + 1, j + 1, k - 1, 7);
//    //
//    //                        _CAS(i - 1, j - 1, k + 0, 7);
//    //                        _CAS(i + 0, j - 1, k + 0, 7);
//    //                        _CAS(i + 1, j - 1, k + 0, 7);
//    //                        _CAS(i - 1, j + 0, k + 0, 7);
//    //                        //   _CAS(i + 0, j + 0, k + 0, 7);
//    //                        _CAS(i + 1, j + 0, k + 0, 7);
//    //                        _CAS(i - 1, j + 1, k + 0, 7);
//    //                        _CAS(i + 0, j + 1, k + 0, 7);
//    //                        _CAS(i + 1, j + 1, k + 0, 7);
//    //
//    //                        _CAS(i - 1, j - 1, k + 1, 7);
//    //                        _CAS(i + 0, j - 1, k + 1, 7);
//    //                        _CAS(i + 1, j - 1, k + 1, 7);
//    //                        _CAS(i - 1, j + 0, k + 1, 7);
//    //                        _CAS(i + 0, j + 0, k + 1, 7);
//    //                        _CAS(i + 1, j + 0, k + 1, 7);
//    //                        _CAS(i - 1, j + 1, k + 1, 7);
//    //                        _CAS(i + 0, j + 1, k + 1, 7);
//    //                        _CAS(i + 1, j + 1, k + 1, 7);
//    //                        break;
//    //                    default:
//    //                        break;
//    //                }
//    //#undef _CAS
//    //            }
//
//    return m_interface_cache_.at(GetIFORM).at(tag_in).at(tag_out);
//}

}  // namespace engine {
}  // namespace simpla {