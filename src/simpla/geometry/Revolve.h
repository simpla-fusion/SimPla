//
// Created by salmon on 16-12-1.
//

#ifndef SIMPLA_REVOLVE_H
#define SIMPLA_REVOLVE_H

#include <simpla/physics/Constants.h>
#include "GeoObject.h"
#include "Polygon.h"

namespace simpla {
namespace geometry {

template <typename TObj>
class Revolve : public GeoObject {
    SP_OBJECT_HEAD(Revolve<TObj>, GeoObject)

   public:
    Revolve(TObj const &obj, int ZAxis = 2) : base_obj(obj), GeoObject() { m_axis_[ZAxis] = 1; }
    Revolve(TObj const &obj, point_type origin, point_type axis)
        : base_obj(obj), m_axis_(axis), m_origin_(origin), GeoObject() {}
    Revolve(this_type const &other)
        : base_obj(other.base_obj), m_origin_(other.m_origin_), m_axis_(other.m_axis_), GeoObject() {}

    ~Revolve() override = default;

    DECLARE_REGISTER_NAME("Revolve");

    std::shared_ptr<data::DataTable> Serialize() const override {
        auto res = data::Serializable::Serialize();
        res->template SetValue<std::string>("Type", "Revolve");
        res->template SetValue<point_type>("Axis", m_axis_);
        res->template SetValue<point_type>("Origin", m_origin_);
        res->Set("2DShape", base_obj.Serialize());
        return res;
    };
    void Deserialize(const std::shared_ptr<data::DataTable> &cfg) override {}

    virtual box_type GetBoundBox() const override { return box_type{{0, 0, 0}, {1, 2, 3}}; };

    bool CheckInside(point_type const &x) const override { return base_obj.CheckInside(MapTo2d(x)); };

    nTuple<Real, 2> MapTo2d(point_type const &x) const {
        nTuple<Real, 2> y{0, 0};
        y[1] = dot(m_axis_, x - m_origin_);                                                     // Z
        y[0] = std::sqrt(dot(x - m_origin_ - y[1] * m_axis_, x - m_origin_ - y[1] * m_axis_));  // R
        return std::move(y);
    };

    point_type m_origin_{0, 0, 0};
    point_type m_axis_{0, 0, 1};

    TObj const &base_obj;
};

template <typename TObj>
std::shared_ptr<GeoObject> revolve(TObj const &obj, int phi_axis = 2) {
    return std::dynamic_pointer_cast<GeoObject>(std::make_shared<Revolve<TObj>>(obj, phi_axis));
}

class RevolveZ : public GeoObject {
    SP_OBJECT_HEAD(RevolveZ, GeoObject)

   public:
    RevolveZ(std::shared_ptr<Polygon<2>> const &obj, int phi_axis, Real phi0, Real phi1, point_type origin = {0, 0, 0})
        : m_origin_(origin), base_obj(obj), m_phi_axe_(phi_axis), m_angle_min_(phi0), m_angle_max_(phi1) {}
    RevolveZ(this_type const &other)
        : base_obj(other.base_obj), m_origin_(other.m_origin_), m_phi_axe_(other.m_phi_axe_) {}
    ~RevolveZ() override = default;

    DECLARE_REGISTER_NAME("RevolveZ");

    std::shared_ptr<data::DataTable> Serialize() const override {
        auto res = data::Serializable::Serialize();
        res->template SetValue<std::string>("Type", "RevolveZ");
        res->template SetValue("Axis", m_phi_axe_);
        res->template SetValue("Origin", m_origin_);
        res->template SetValue("Phi", nTuple<Real, 2>{m_angle_min_, m_angle_max_});

        res->Set("2DShape", base_obj->Serialize());
        return res;
    };
    void Deserialize(const std::shared_ptr<data::DataTable> &cfg) override {}

    box_type GetBoundBox() const override {
        nTuple<Real, 2> lo, hi;
        std::tie(lo, hi) = base_obj->GetBoundBox();
        box_type res;
        std::get<0>(res)[m_phi_axe_] = m_angle_min_;
        std::get<1>(res)[m_phi_axe_] = m_angle_max_;

        std::get<0>(res)[(m_phi_axe_ + 1) % 3] = lo[0];
        std::get<1>(res)[(m_phi_axe_ + 1) % 3] = hi[0];

        std::get<0>(res)[(m_phi_axe_ + 2) % 3] = lo[1];
        std::get<1>(res)[(m_phi_axe_ + 2) % 3] = hi[1];
        return res;
    };

    bool CheckInside(point_type const &x) const override {
        return ((x[m_phi_axe_] >= m_angle_min_) && (x[m_phi_axe_] < m_angle_max_)) &&
               base_obj->check_inside(x[(m_phi_axe_ + 1) % 3], x[(m_phi_axe_ + 2) % 3]);
    };

    nTuple<Real, 2> MapTo2d(point_type const &x) const {
        return nTuple<Real, 2>{x[(m_phi_axe_ + 1) % 3] - m_origin_[(m_phi_axe_ + 1) % 3],
                               x[(m_phi_axe_ + 2) % 3] - m_origin_[(m_phi_axe_ + 2) % 3]};
    };

    point_type m_origin_{0, 0, 0};
    Real m_angle_min_ = 0, m_angle_max_ = TWOPI;
    int m_phi_axe_ = 2;

    std::shared_ptr<Polygon<2>> base_obj;
};
}
}
#endif  // SIMPLA_REVOLVE_H
