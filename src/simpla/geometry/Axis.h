//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_AXIS_H
#define SIMPLA_AXIS_H

#include <simpla/data/DataNode.h>
namespace simpla {
template <typename, int...>
struct nTuple;
namespace geometry {
struct Plane;
struct Line;
struct Axis {
    Axis() = default;
    Axis(Axis const &other) : o(other.m_origin_), m_axis_(other.m_axis_){};
    ~Axis() = default;
    Axis(std::initializer_list<std::initializer_list<Real>> const &v) {
        m_axis_[0] = point_type{*v.begin()};
        m_axis_[1] = point_type{*(v.begin() + 1)};
        m_axis_[2] = point_type{*(v.begin() + 2)};
    }

    Axis(point_type const &origin, vector_type const &x_axis) : m_origin_(origin) {
        m_axis_[0] = x_axis;
        m_axis_[1] = 0;
        m_axis_[2] = cross(m_axis_[0], m_axis_[1]);
    }
    Axis(point_type const &origin, vector_type const &x_axis, vector_type const &y_axis) : m_origin_(origin) {
        m_axis_[0] = x_axis;
        m_axis_[1] = y_axis;
        m_axis_[2] = cross(x_axis, y_axis);
    }
    Axis(point_type const &origin, vector_type const &x_axis, vector_type const &y_axis, vector_type const &z_axis)
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
    void Move(const point_type &p);

    virtual point_type xyz(point_type const &uvw_) const { return o + uvw_[0] * x + uvw_[1] * y + uvw_[2] * z; }
    virtual point_type uvw(point_type const &xyz_) const {
        return point_type{dot(xyz_ - o, x), dot(xyz_ - o, y), dot(xyz_ - o, z)};
    }
    point_type xyz(Real u, Real v = 0, Real w = 0) const { return xyz(point_type{u, v, w}); }
    point_type uvw(Real x0, Real x1 = 0, Real x2 = 0) const { return uvw(point_type{x0, x1, x2}); }
    point_type Coordinates(Real u, Real v = 0, Real w = 0) const { return xyz(point_type{u, v, w}); }

    std::shared_ptr<Plane> GetPlane(int) const;
    std::shared_ptr<Plane> GetPlaneXY() const;
    std::shared_ptr<Plane> GetPlaneYZ() const;
    std::shared_ptr<Plane> GetPlaneZX() const;
    std::shared_ptr<Line> GetAxe(int) const;
    std::shared_ptr<Line> GetPlaneX() const;
    std::shared_ptr<Line> GetPlaneY() const;
    std::shared_ptr<Line> GetPlaneZ() const;

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
