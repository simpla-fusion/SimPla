//
// Created by salmon on 17-2-17.
//
#include "Chart.h"
#include <simpla/utilities/Log.h>
#include <simpla/utilities/SingletonHolder.h>
#include <set>
#include "Attribute.h"

namespace simpla {
namespace engine {
struct Chart::pimpl_s {
    int m_level_ = 0;
    point_type m_origin_{0, 0, 0};
    point_type m_dx_{1, 1, 1};
    point_type m_inv_dx_{1, 1, 1};
};

Chart::Chart() : m_pimpl_(new pimpl_s) {}
Chart::~Chart() {}

void Chart::SetOrigin(point_type const &x0) { m_pimpl_->m_origin_ = x0; }
void Chart::SetDx(point_type const &dx) { m_pimpl_->m_dx_ = dx; }
point_type const &Chart::GetOrigin() const { return m_pimpl_->m_origin_; }
point_type const &Chart::GetDx() const { return m_pimpl_->m_dx_; }
std::shared_ptr<data::DataTable> Chart::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    //    p->SetValue<std::string>("Type", "CartesianGeometry");
    p->SetValue("Origin", m_pimpl_->m_origin_);
    p->SetValue("Dx", m_pimpl_->m_dx_);
    return p;
};
void Chart::Deserialize(std::shared_ptr<data::DataTable> d) {
    m_pimpl_->m_dx_ = d->GetValue<point_type>("Dx");
    m_pimpl_->m_origin_ = d->GetValue<point_type>("Origin");
    m_pimpl_->m_inv_dx_ = 1.0 / m_pimpl_->m_dx_;
}

point_type Chart::map(point_type const &x) const {
    point_type res;
    res = (x - m_pimpl_->m_origin_) * m_pimpl_->m_inv_dx_;
    return std::move(res);
}
point_type Chart::inv_map(point_type const &x) const {
    point_type res;
    res = x * m_pimpl_->m_dx_ + m_pimpl_->m_origin_;
    return std::move(res);
}
point_type Chart::inv_map(index_tuple const &x) const {
    point_type res;
    res = x * m_pimpl_->m_dx_ + m_pimpl_->m_origin_;
    return std::move(res);
}

box_type Chart::map(box_type const &b) const { return std::make_tuple(map(std::get<0>(b)), map(std::get<1>(b))); };
box_type Chart::inv_map(box_type const &b) const {
    return std::make_tuple(inv_map(std::get<0>(b)), inv_map(std::get<1>(b)));
};
box_type Chart::inv_map(index_box_type const &b) const {
    return std::make_tuple(inv_map(std::get<0>(b)), inv_map(std::get<1>(b)));
};
// int Chart::GetLevel() const { return m_pimpl_->m_level_; }
//
// struct ChartFactory {
//    std::map<std::string, std::function<Chart *()>> m_mesh_factory_;
//};
//
// bool Chart::RegisterCreator(std::string const &k, std::function<Chart *()> const &fun) {
//    auto res = SingletonHolder<ChartFactory>::instance().m_mesh_factory_.emplace(k, fun).second;
//    if (res) { LOGGER << "MeshBase Creator [ " << k << " ] is registered!" << std::endl; }
//    return res;
//}
// Chart *Chart::Create(std::string const &k) {
//    Chart *res = nullptr;
//
//    try {
//        res = SingletonHolder<ChartFactory>::instance().m_mesh_factory_.at(k)();
//    } catch (std::out_of_range const &) { RUNTIME_ERROR << "Missing mesh creator  [" << k << "]!" << std::endl; }
//
//    if (res != nullptr) { LOGGER << "MeshBase [" << k << "] is created!" << std::endl; }
//    return res;
//};
//
// Chart *Chart::Create(std::shared_ptr<data::DataTable> const &config) {
//    auto res = Create(config->GetValue<std::string>("name", ""));
//    res->db()->PushPatch(*config);
//    return res;
//}

}  // namespace engine{
}  // namespace simpla{
