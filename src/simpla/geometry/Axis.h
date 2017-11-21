//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_AXIS_H
#define SIMPLA_AXIS_H

#include <simpla/data/Configurable.h>
#include <simpla/data/DataEntry.h>
#include <simpla/data/Serializable.h>
namespace simpla {
namespace geometry {
namespace detail {

inline vector_type make_perp1(vector_type const &v) {
    vector_type res{0, 0, 0};
    if (dot(v, v) < SP_EPSILON) {
        RUNTIME_ERROR << "no perpendicular vector for zero ";
    } else if (std::abs(v[0]) < SP_EPSILON) {
        res[0] = 1;
    } else if (std::abs(v[1]) < SP_EPSILON) {
        res[1] = 1;
    } else if (std::abs(v[2]) < SP_EPSILON) {
        res[2] = 1;
    } else {
        res[0] = 1;
        res[1] = 1;
        res[2] = -(v[0] + v[1]) / v[2];
    }
    return normal(res);
}
inline vector_type make_perp2(vector_type const &v) { return normal(cross(v, make_perp1(v))); }
}  // namespace detail{

struct Axis : public data::Serializable, public data::Configurable {
    SP_SERIALIZABLE_HEAD(data::Serializable, Axis)

    Axis() = default;
    Axis(Axis const &other) = default;
    Axis(std::initializer_list<std::initializer_list<Real>> const &list) : m_Axis_(list) {}
    Axis(point_type origin, vector_type const &x_axis, vector_type const &y_axis, vector_type const &z_axis)
        : m_Origin_(std::move(origin)), m_Axis_{x_axis, y_axis, z_axis} {}
    Axis(point_type const &origin, vector_type const &x_axis, vector_type const &y_axis)
        : Axis(origin, x_axis, y_axis, cross(x_axis, y_axis)) {}
    Axis(point_type const &origin, vector_type const &x_axis)
        : Axis(origin, x_axis, detail::make_perp1(x_axis), detail::make_perp2(x_axis)) {}
    ~Axis() override = default;

    void swap(Axis &other) {
        std::swap(m_Origin_, other.m_Origin_);
        std::swap(m_Axis_, other.m_Axis_);
    };

    Axis &operator=(Axis const &other) {
        Axis(other).swap(*this);
        return *this;
    };

    virtual std::shared_ptr<Axis> Copy() const { return std::make_shared<Axis>(*this); };

    template <typename... Args>
    void SetUp(point_type const &origin, Args &&... args) {
        SetOrigin(origin);
        SetAxis(std::forward<Args>(args)...);
    }

    void SetAxis(int n, vector_type const &v) { m_Axis_[n] = v; }
    vector_type GetAxis(int n) const { return m_Axis_[n]; }
    vector_type const &GetDirection(int n) const { return m_Axis_[n]; }

    void SetAxis(point_type const &x_axis, point_type const &y_axis, point_type const &z_axis) {
        m_Axis_[0] = x_axis;
        m_Axis_[1] = y_axis;
        m_Axis_[2] = z_axis;
    }

    void Mirror(const point_type &p);
    void Mirror(const Axis &a1);
    void Rotate(const Axis &a1, Real angle);
    void Scale(Real s, int dir = -1);
    void Translate(const vector_type &v);
    void Move(const point_type &p);
    Axis Moved(const point_type &p) const;

    virtual point_type xyz(point_type const &uvw_) const { return o + uvw_[0] * x + uvw_[1] * y + uvw_[2] * z; }
    virtual point_type uvw(point_type const &xyz_) const {
        return point_type{dot(xyz_ - o, x), dot(xyz_ - o, y), dot(xyz_ - o, z)};
    }
    point_type xyz(Real u, Real v = 0, Real w = 0) const { return xyz(point_type{u, v, w}); }
    point_type uvw(Real x0, Real x1 = 0, Real x2 = 0) const { return uvw(point_type{x0, x1, x2}); }
    point_type Coordinates(Real u, Real v = 0, Real w = 0) const { return xyz(point_type{u, v, w}); }

    SP_PROPERTY(point_type, Origin) = {0, 0, 0};
    SP_PROPERTY(matrix_type, Axis) = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

   private:
   public:
    point_type const &o = m_Origin_;
    vector_type const &x = m_Axis_[0];
    vector_type const &y = m_Axis_[1];
    vector_type const &z = m_Axis_[2];
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_AXIS_H
