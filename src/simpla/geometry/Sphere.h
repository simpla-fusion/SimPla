//
// Created by salmon on 17-7-21.
//

#ifndef SIMPLA_SPHERE_H
#define SIMPLA_SPHERE_H

#include "simpla/SIMPLA_config.h"

#include "GeoObject.h"

namespace simpla {
namespace geometry {

struct Sphere : public GeoObject {
    SP_OBJECT_HEAD(Sphere, GeoObject)
    DECLARE_REGISTER_NAME(Sphere);
    Real m_radius_ = 1;
    point_type m_origin_{0, 0, 0};

    Sphere() = default;
    Sphere(Real r, point_type o) : m_radius_(r), m_origin_(std::move(o)) {}

    ~Sphere() override = default;

    std::shared_ptr<data::DataTable> Serialize() const override {
        auto p = std::make_shared<data::DataTable>();
        p->SetValue<std::string>("Type", GetRegisterName());
        p->SetValue("Origin", m_origin_);
        p->SetValue("Radius", m_radius_);

        return p;
    };
    void Deserialize(std::shared_ptr<data::DataTable> const &cfg) override {
        m_origin_ = cfg->GetValue("Origin", m_origin_);
        m_radius_ = cfg->GetValue("Radius", m_radius_);
    }

    box_type GetBoundBox() const override {
        box_type b;
        std::get<0>(b) = m_origin_ - m_radius_;
        std::get<1>(b) = m_origin_ + m_radius_;
        return std::move(b);
    };

    bool CheckInside(point_type const &x) const override {
        return (x - m_origin_) * (x - m_origin_) - m_radius_ * m_radius_ < 0;
    }
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_SPHERE_H
