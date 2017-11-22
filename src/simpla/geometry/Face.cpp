//
// Created by salmon on 17-11-14.
//

#include "Face.h"
namespace simpla {
namespace geometry {

Face::Face(Axis const &axis, std::shared_ptr<const gSurface> const &surface, Real u_max, Real v_max)
    : Face(axis, surface, 0, 0, u_max, v_max) {}
Face::Face(Axis const &axis, std::shared_ptr<const gSurface> const &surface, Real u_min, Real u_max, Real v_min,
           Real v_max)
    : Face(axis, surface, std::tuple<point2d_type, point2d_type>{{u_min, v_min}, {u_max, v_max}}){};
Face::Face(Axis const &axis, std::shared_ptr<const gSurface> const &surface,
           std::tuple<point2d_type, point2d_type> const &range)
    : GeoObject(axis), m_surface_(surface), m_range_{range} {};

void Face::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_surface_ = gSurface::CreateAs<gSurface>(cfg->Get("gSurface"));
    m_range_ = cfg->GetValue("ParameterRange", m_range_);
};
std::shared_ptr<simpla::data::DataEntry> Face::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("gSurface", m_surface_->Serialize());
    res->SetValue("ParameterRange", m_range_);
    return res;
};
void Face::SetSurface(std::shared_ptr<const gSurface> const &s) { m_surface_ = s; }
std::shared_ptr<const gSurface> Face::GetSurface() const { return m_surface_; }
void Face::SetParameterRange(std::tuple<point2d_type, point2d_type> const &r) { m_range_ = r; }
std::tuple<point2d_type, point2d_type> const &Face::GetParameterRange() const { return m_range_; };
}  // namespace geometry
}  // namespace simpla