//
// Created by salmon on 17-10-23.
//
#include "Axis.h"
#include <simpla/data/Serializable.h>
namespace simpla {
namespace geometry {
Axis::Axis(Axis const &other) : data::Configurable(other), m_Origin_(other.m_Origin_), m_axis_(other.m_axis_) {}
void Axis::Mirror(const point_type &p) { UNIMPLEMENTED; }
void Axis::Mirror(const Axis &a1) { UNIMPLEMENTED; }
void Axis::Rotate(const Axis &a1, Real angle) { UNIMPLEMENTED; }
void Axis::Rotate(vector_type const &u, Real angle) {
    Real ux = u[0];
    Real uy = u[1];
    Real uz = u[2];
    Real cosA = std::cos(angle);
    Real sinA = std::sin(angle);
    Real i_u2 = 1.0 / dot(u, u);
    /** @ref https://en.wikipedia.org/wiki/Rotation_matrix
     *  Rotation matrix from axis and angle
     *  \f[
     *  {\displaystyle R={\begin{bmatrix}\cos \theta +u_{x}^{2}\left(1-\cos \theta \right)&u_{x}u_{y}\left(1-\cos
     * \theta
     * \right)-u_{z}\sin \theta &u_{x}u_{z}\left(1-\cos \theta \right)+u_{y}\sin \theta \\u_{y}u_{x}\left(1-\cos
     * \theta
     * \right)+u_{z}\sin \theta &\cos \theta +u_{y}^{2}\left(1-\cos \theta \right)&u_{y}u_{z}\left(1-\cos \theta
     * \right)-u_{x}\sin \theta \\u_{z}u_{x}\left(1-\cos \theta \right)-u_{y}\sin \theta &u_{z}u_{y}\left(1-\cos
     * \theta
     * \right)+u_{x}\sin \theta &\cos \theta +u_{z}^{2}\left(1-\cos \theta \right)\end{bmatrix}}.}
     *  \f]
     */
    //    Matrix<Real, 3, 3> R = {
    //        {cosA + ux * ux * (1 - cosA), ux * uy * (1 - cosA) - uz * sinA, ux * uz * (1 - cosA) + uy * sinA},
    //        {uy * ux * (1 - cosA) + uz * sinA, cosA + uy * uy * (1 - cosA), uy * uz * (1 - cosA) - ux * sinA},
    //        {uz * ux * (1 - cosA) - uy * sinA, uz * uy * (1 - cosA) + ux * sinA, cosA + uz * uz * (1 - cosA)}};
    //    R /= ux * ux + uy * uy + uz * uz;
    //    m_axis_[0] = point_type{dot(m_axis_[0], R[0]), dot(m_axis_[0], R[1]), dot(m_axis_[0], R[2])};
    //    m_axis_[1] = point_type{dot(m_axis_[1], R[0]), dot(m_axis_[1], R[1]), dot(m_axis_[1], R[2])};
    //    m_axis_[2] = point_type{dot(m_axis_[2], R[0]), dot(m_axis_[2], R[1]), dot(m_axis_[2], R[2])};

    m_axis_[0] = (cosA * m_axis_[0] + sinA * cross(u, m_axis_[0]) + (1 - cosA) * dot(u, m_axis_[0]) * u) * i_u2;
    m_axis_[1] = (cosA * m_axis_[1] + sinA * cross(u, m_axis_[1]) + (1 - cosA) * dot(u, m_axis_[1]) * u) * i_u2;
    m_axis_[2] = (cosA * m_axis_[2] + sinA * cross(u, m_axis_[2]) + (1 - cosA) * dot(u, m_axis_[2]) * u) * i_u2;
}

void Axis::Scale(Real s, int dir) {
    if (dir < 0) {
        m_axis_ *= s;
    } else {
        m_axis_[dir % 3] *= s;
    }
}
void Axis::Translate(const vector_type &v) { m_Origin_ += v; }
void Axis::Translate(Axis const &other) {
    m_Origin_ += other.m_Origin_;
    matrix_type t_axis = m_axis_;
    m_axis_[0] = other.m_axis_[0][0] * t_axis[0] + other.m_axis_[0][1] * t_axis[1] + other.m_axis_[0][2] * t_axis[2];
    m_axis_[1] = other.m_axis_[1][0] * t_axis[1] + other.m_axis_[1][1] * t_axis[1] + other.m_axis_[1][2] * t_axis[2];
    m_axis_[2] = other.m_axis_[2][0] * t_axis[2] + other.m_axis_[2][1] * t_axis[1] + other.m_axis_[2][2] * t_axis[2];
}
void Axis::Shift(const point_type &o_uvw) { m_Origin_ += xyz(o_uvw); }
Axis Axis::Shifted(const point_type &o_uvw) const {
    Axis res(*this);
    res.Shift(o_uvw);
    return std::move(res);
}
void Axis::Move(const point_type &o_xyz) { m_Origin_ = o_xyz; }
Axis Axis::Moved(const point_type &o_xyz) const {
    Axis res(*this);
    res.Move(o_xyz);
    return std::move(res);
}

//
// std::shared_ptr<spPlane> Axis::GetPlane(int n) const {
//    return spPlane::New(Axis{o, m_axis_[(n + 1) % 3], m_axis_[(n + 2) % 3]});
//}
// std::shared_ptr<spPlane> Axis::GetPlaneXY() const { return GetPlane(2); }
// std::shared_ptr<spPlane> Axis::GetPlaneYZ() const { return GetPlane(1); }
// std::shared_ptr<spPlane> Axis::GetPlaneZX() const { return GetPlane(0); }
// std::shared_ptr<spLine> Axis::GetAxe(int n) const { return spLine::New(Axis{o, m_axis_[n]}); }
// std::shared_ptr<spLine> Axis::GetPlaneX() const { return GetAxe(0); }
// std::shared_ptr<spLine> Axis::GetPlaneY() const { return GetAxe(0); }
// std::shared_ptr<spLine> Axis::GetPlaneZ() const { return GetAxe(0); }

}  // namespace geometry{
}  // namespace simpla{