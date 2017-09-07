//
// Created by salmon on 16-12-1.
//

#ifndef SIMPLA_REVOLVE_H
#define SIMPLA_REVOLVE_H

#include "simpla/SIMPLA_config.h"

#include "simpla/utilities/Constants.h"

#include "GeoObject.h"
#include "Polygon.h"

namespace simpla {
namespace geometry {
template <int NDIMS>
class Polygon;
template <typename TObj>
class Revolve : public GeoObject {
    SP_OBJECT_HEAD(Revolve<TObj>, GeoObject)
   protected:
    Revolve(TObj const &obj, int ZAxis = 2) : base_obj(obj) { m_axis_[ZAxis] = 1; }
    Revolve(TObj const &obj, point_type origin, point_type axis) : base_obj(obj), m_axis_(axis), m_origin_(origin) {}

   public:
    virtual box_type GetBoundingBox() const override { return box_type{{0, 0, 0}, {1, 2, 3}}; };

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
std::shared_ptr<data::DataNode> Revolve<TObj>::Serialize() const {
    auto tdb = base_type::Serialize();

    tdb->SetValue("Axis", m_axis_);
    tdb->SetValue("Origin", m_origin_);
    tdb->Set("2DShape", base_obj.Pack());

    return tdb;
};
template <typename TObj>
void Revolve<TObj>::Deserialize(std::shared_ptr<data::DataNode>  const&cfg) {
    base_type::Deserialize(cfg);
}

template <typename TObj>
std::shared_ptr<GeoObject> revolve(TObj const &obj, int phi_axis = 2) {
    return std::dynamic_pointer_cast<GeoObject>(Revolve<TObj>::New(obj, phi_axis));
}

class RevolveZ : public GeoObject {
    SP_OBJECT_HEAD(RevolveZ, GeoObject)
   protected:
    RevolveZ(std::shared_ptr<Polygon<2>> const &obj, int phi_axis, Real phi0, Real phi1, point_type origin = {0, 0, 0})
        : m_origin_(origin), base_obj(obj), m_phi_axe_(phi_axis), m_angle_min_(phi0), m_angle_max_(phi1) {}

   public:
    box_type GetBoundingBox() const override {
        nTuple<Real, 2> lo, hi;
        std::tie(lo, hi) = base_obj->GetBoundingBox();
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
