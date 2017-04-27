//
// Created by salmon on 16-12-1.
//

#ifndef SIMPLA_REVOLVE_H
#define SIMPLA_REVOLVE_H

#include "GeoObject.h"
#include "Polygon.h"

namespace simpla {
namespace geometry {

template <typename TObj>
class Revolve : public GeoObject {
    SP_OBJECT_HEAD(Revolve<TObj>, GeoObject)

   public:
    Revolve(TObj const &obj, int ZAxis = 2) : base_obj(obj) { m_axis_[ZAxis] = 1; }
    Revolve(TObj const &obj, point_type origin, point_type axis) : base_obj(obj), m_axis_(axis), m_origin_(origin) {}
    Revolve(this_type const &other) : base_obj(other.base_obj), m_origin_(other.m_origin_), m_axis_(other.m_axis_) {}

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
    void Deserialize(const std::shared_ptr<data::DataTable> &t) override {}

    int check_inside(point_type const &x) const { return base_obj.check_inside(MapTo2d(x)); };
    nTuple<Real, 2> MapTo2d(point_type const &x) const {
        nTuple<Real, 2> y{0, 0};
        y[1] = vec_dot(m_axis_, x - m_origin_);                                                     // Z
        y[0] = std::sqrt(vec_dot(x - m_origin_ - y[1] * m_axis_, x - m_origin_ - y[1] * m_axis_));  // R
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
}
}
#endif  // SIMPLA_REVOLVE_H
