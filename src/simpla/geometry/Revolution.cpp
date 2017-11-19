//
// Created by salmon on 17-10-24.
//

#include "Revolution.h"
#include "Shell.h"
#include "Solid.h"
#include "Wire.h"
namespace simpla {
namespace geometry {

RevolutionShell::RevolutionShell(Axis const &axis, std::shared_ptr<const Wire> const &g, Real min_angle, Real max_angle)
    : Shell(axis), m_basis_obj_(g), m_MinAngle_(min_angle), m_MaxAngle_(max_angle) {}
RevolutionShell::RevolutionShell(Axis const &axis, std::shared_ptr<const Wire> const &g, Real angle)
    : RevolutionShell(axis, g, 0, angle) {}
void RevolutionShell::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_basis_obj_ = Wire::New(cfg->Get("Wire"));
}
std::shared_ptr<simpla::data::DataEntry> RevolutionShell::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("Wire", m_basis_obj_->Serialize());
    return res;
}

RevolutionFace::RevolutionFace(Axis const &axis, std::shared_ptr<const Edge> const &g, Real min_angle, Real max_angle)
    : Face(axis, nullptr), m_basis_obj_(g), m_MinAngle_(min_angle), m_MaxAngle_(max_angle) {}
RevolutionFace::RevolutionFace(Axis const &axis, std::shared_ptr<const Edge> const &g, Real angle)
    : RevolutionFace(axis, g, 0, angle) {}

void RevolutionFace::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_basis_obj_ = Edge::New(cfg->Get("Edge"));
}
std::shared_ptr<simpla::data::DataEntry> RevolutionFace::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("Edge", m_basis_obj_->Serialize());
    return res;
}

// point_type Revolution::xyz(Real u, Real v, Real w) const {
//    auto b = std::dynamic_pointer_cast<const PrimitiveShape>(GetBasisObject());
//    ASSERT(b != nullptr);
//    point_type p = b->xyz(u, v, 0);
//    Real sinw = std::sin(w);
//    Real cosw = std::cos(w);
//    return m_axis_.xyz(p[0] * cosw - p[1] * sinw, p[0] * sinw + p[1] * cosw, p[2]);
//};
RevolutionSolid::RevolutionSolid(Axis const &axis, std::shared_ptr<const Face> const &f, Real min_angle, Real max_angle)
    : Solid(axis), m_basis_obj_(f), m_MinAngle_(min_angle), m_MaxAngle_(max_angle) {}
RevolutionSolid::RevolutionSolid(Axis const &axis, std::shared_ptr<const Face> const &f, Real angle)
    : RevolutionSolid(axis, f, 0, angle) {}
void RevolutionSolid::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_basis_obj_ = Face::New(cfg->Get("Face"));
}
std::shared_ptr<simpla::data::DataEntry> RevolutionSolid::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("Face", m_basis_obj_->Serialize());
    return res;
}
std::shared_ptr<Shell> MakeRevolution(Axis const &axis, std::shared_ptr<const Wire> const &w, Real angle0,
                                      Real angle1) {
    return RevolutionShell::New(axis, w, angle0, angle1);
}
std::shared_ptr<Face> MakeRevolution(Axis const &axis, std::shared_ptr<const Edge> const &e, Real angle0, Real angle1) {
    return RevolutionFace::New(axis, e, angle0, angle1);
}
std::shared_ptr<Solid> MakeRevolution(Axis const &axis, std::shared_ptr<const Face> const &f, Real angle0,
                                      Real angle1) {
    return RevolutionSolid::New(axis, f, angle0, angle1);
}

}  // namespace geometry{
}  // namespace simpla{