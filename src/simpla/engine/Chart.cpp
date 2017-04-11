//
// Created by salmon on 17-2-17.
//
#include "Chart.h"
#include <simpla/design_pattern/SingletonHolder.h>
#include <simpla/toolbox/Log.h>
#include <set>

namespace simpla {
namespace engine {
struct Chart::pimpl_s {
    int m_level_ = 0;
    point_type m_origin_{0, 0, 0};
    point_type m_dx_{1, 1, 1};
};

Chart::Chart() : m_pimpl_(new pimpl_s) {}
Chart::~Chart() {}
point_type const &Chart::GetOrigin() const { return m_pimpl_->m_origin_; }
point_type const &Chart::GetDx() const { return m_pimpl_->m_dx_; }
std::shared_ptr<data::DataTable> Chart::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue<std::string>("Type", "CartesianGeometry");
    p->SetValue("Origin", m_pimpl_->m_origin_);
    p->SetValue("Dx", m_pimpl_->m_dx_);
    return p;
};
void Chart::Deserialize(std::shared_ptr<data::DataTable> const &d) {
    m_pimpl_->m_dx_ = d->GetValue<point_type>("Dx");
    m_pimpl_->m_origin_ = d->GetValue<point_type>("Origin");
}

// int Chart::GetLevel() const { return m_pimpl_->m_level_; }
//
// struct ChartFactory {
//    std::map<std::string, std::function<Chart *()>> m_mesh_factory_;
//};
//
// bool Chart::RegisterCreator(std::string const &k, std::function<Chart *()> const &fun) {
//    auto res = SingletonHolder<ChartFactory>::instance().m_mesh_factory_.emplace(k, fun).second;
//    if (res) { LOGGER << "Mesh Creator [ " << k << " ] is registered!" << std::endl; }
//    return res;
//}
// Chart *Chart::Create(std::string const &k) {
//    Chart *res = nullptr;
//
//    try {
//        res = SingletonHolder<ChartFactory>::instance().m_mesh_factory_.at(k)();
//    } catch (std::out_of_range const &) { RUNTIME_ERROR << "Missing mesh creator  [" << k << "]!" << std::endl; }
//
//    if (res != nullptr) { LOGGER << "Mesh [" << k << "] is created!" << std::endl; }
//    return res;
//};
//
// Chart *Chart::Create(std::shared_ptr<data::DataTable> const &config) {
//    auto res = Create(config->GetValue<std::string>("name", ""));
//    res->db()->Set(*config);
//    return res;
//}

}  // namespace engine{
}  // namespace simpla{
