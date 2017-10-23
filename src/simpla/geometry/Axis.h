//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_AXIS_H
#define SIMPLA_AXIS_H

#include <simpla/data/DataNode.h>
namespace simpla {
namespace geometry {
struct Axis {
    Axis() = default;
    Axis(Axis const &other) : o(other.o), m_axis_(other.m_axis_){};
    ~Axis() = default;
    Axis(point_type const &origin, point_type const &x_axis) : m_origin_(origin) {
        m_axis_[0] = x_axis;
        m_axis_[1] = 0;
        m_axis_[2] = cross(m_axis_[0], m_axis_[1]);
    }
    Axis(point_type const &origin, point_type const &x_axis, point_type const &y_axis) : m_origin_(origin) {
        m_axis_[0] = x_axis;
        m_axis_[1] = y_axis;
        m_axis_[2] = cross(x_axis, y_axis);
    }
    Axis(point_type const &origin, point_type const &x_axis, point_type const &y_axis, point_type const &z_axis)
        : m_origin_(origin) {
        m_axis_[0] = x_axis;
        m_axis_[1] = y_axis;
        m_axis_[2] = z_axis;
    }
    Axis &operator=(Axis const &other) {
        m_origin_ = other.m_origin_;
        m_axis_ = other.m_axis_;
        return *this;
    };
    template <typename... Args>
    static std::shared_ptr<Axis> New(point_type const &origin, point_type const &x_axis) {
        return std::make_shared<Axis>(origin, x_axis);
    };
    template <typename... Args>
    static std::shared_ptr<Axis> New(point_type const &origin, point_type const &x_axis, point_type const &y_axis) {
        return std::make_shared<Axis>(origin, x_axis, y_axis);
    };
    template <typename... Args>
    static std::shared_ptr<Axis> New(point_type const &origin, point_type const &x_axis, point_type const &y_axis,
                                     point_type const &z_axis) {
        return std::make_shared<Axis>(origin, x_axis, y_axis, z_axis);
    };
    static std::shared_ptr<Axis> New(std::shared_ptr<data::DataNode> const &cfg) {
        std::shared_ptr<Axis> res(new Axis);
        res->Deserialize(cfg);
        return res;
    };
    virtual std::shared_ptr<Axis> Copy() const { return std::make_shared<Axis>(*this); };

    void Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg);
    std::shared_ptr<simpla::data::DataNode> Serialize() const;

    template <typename... Args>
    void SetUp(point_type const &origin, Args &&... args) {
        SetOrigin(origin);
        SetMatrix(std::forward<Args>(args)...);
    }

    void SetOrigin(point_type const &a) { m_origin_ = a; }
    point_type const &GetOrigin() const { return m_origin_; }

    void SetMatrix(point_type const &x_axis, point_type const &y_axis, point_type const &z_axis) {
        m_axis_[0] = x_axis;
        m_axis_[1] = y_axis;
        m_axis_[2] = z_axis;
    }
    void SetMatrix(nTuple<Real, 3, 3> const &a) { m_axis_ = a; }
    nTuple<Real, 3, 3> const &GetMatrix() const { return m_axis_; }

    void Mirror(const point_type &p);
    void Mirror(const Axis &a1);
    void Rotate(const Axis &a1, Real angle);
    void Scale(Real s, int dir = -1);
    void Translate(const vector_type &v);

    virtual point_type Coordinates(Real u, Real v = 0, Real w = 0) const { return o + u * x + v * y + v * z; }

   private:
    point_type m_origin_{0, 0, 0};
    nTuple<Real, 3, 3> m_axis_{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

   public:
    point_type const &o = m_origin_;
    vector_type const &x = m_axis_[0];
    vector_type const &y = m_axis_[1];
    vector_type const &z = m_axis_[2];
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_AXIS_H
