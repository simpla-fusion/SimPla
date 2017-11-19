//
// Created by salmon on 17-11-14.
//

#include "Face.h"
namespace simpla {
namespace geometry {
Face::Face() = default;
Face::~Face() = default;
Face::Face(Face const &other) = default;
Face::Face(Axis const &axis, std::shared_ptr<const Surface> const &surface, Real u_max, Real v_max)
    : Face(axis, surface, 0, 0, u_max, v_max) {}
Face::Face(Axis const &axis, std::shared_ptr<const Surface> const &surface, Real u_min, Real u_max, Real v_min,
           Real v_max)
    : Face(axis, surface, std::tuple<point2d_type, point2d_type>{{u_min, v_min}, {u_max, v_max}}){};
Face::Face(Axis const &axis, std::shared_ptr<const Surface> const &surface,
           std::tuple<point2d_type, point2d_type> const &range)
    : GeoObject(axis), m_surface_(surface), m_range_{range} {};

void Face::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_surface_ = Surface::New(cfg->Get("Surface"));
    m_range_ = cfg->GetValue("ParameterRange", m_range_);
};
std::shared_ptr<simpla::data::DataEntry> Face::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("Surface", m_surface_->Serialize());
    res->SetValue("ParameterRange", m_range_);
    return res;
};
}  // namespace geometry
}  // namespace simpla