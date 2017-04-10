//
// Created by salmon on 17-2-17.
//
#include "Chart.h"
#include <set>
#include <simpla/toolbox/Log.h>
#include <simpla/design_pattern/SingletonHolder.h>

namespace simpla {
namespace engine {
struct Chart::pimpl_s {
    int m_level_ = 0;
    point_type m_origin_;
    point_type m_dx_;
};

Chart::Chart() : m_pimpl_(new pimpl_s) {}
Chart::~Chart() {}
point_type const& Chart::GetOrigin() const { return m_pimpl_->m_origin_; }
point_type const& Chart::GetDx() const { return m_pimpl_->m_dx_; }
int Chart::GetLevel() const { return m_pimpl_->m_level_; }


struct ChartFactory {
    std::map<std::string, std::function<Chart *()>> m_mesh_factory_;
};

bool Chart::RegisterCreator(std::string const &k, std::function<Chart *()> const &fun) {
    auto res = SingletonHolder<ChartFactory>::instance().m_mesh_factory_.emplace(k, fun).second;
    if (res) { LOGGER << "Mesh Creator [ " << k << " ] is registered!" << std::endl; }
    return res;
}
Chart *Chart::Create(std::shared_ptr<data::DataTable> const &config) {
    Chart *res = nullptr;
    try {
        if (config != nullptr) {
            res = SingletonHolder<ChartFactory>::instance().m_mesh_factory_.at(
                    config->GetValue<std::string>("name", ""))();
            res->db() = config;
        }

    } catch (std::out_of_range const &) {
        RUNTIME_ERROR << "Mesh creator  [] is missing!" << std::endl;
        return nullptr;
    }
    if (res != nullptr) { LOGGER << "Mesh [" << res->name() << "] is created!" << std::endl; }
    return res;
}

}  // namespace engine{
}  // namespace simpla{
