//
// Created by salmon on 17-10-23.
//
#include "ParametricSurface.h"
namespace simpla {
namespace geometry {

void ParametricSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_origin_ = cfg->GetValue("Origin", m_origin_);
    m_x_axis_ = cfg->GetValue("XAxis", m_x_axis_);
    m_y_axis_ = cfg->GetValue("YAxis", m_y_axis_);
    m_z_axis_ = cfg->GetValue("ZAxis", m_z_axis_);
}
std::shared_ptr<simpla::data::DataNode> ParametricSurface::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Origin", m_origin_);
    res->SetValue("XAxis", m_x_axis_);
    res->SetValue("YAxis", m_y_axis_);
    res->SetValue("ZAxis", m_z_axis_);
    return res;
}

}  // namespace geometry{
}  // namespace simpla{