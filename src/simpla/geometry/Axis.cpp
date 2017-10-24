//
// Created by salmon on 17-10-23.
//
#include "Axis.h"
namespace simpla {
namespace geometry {

void Axis::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    m_origin_ = cfg->GetValue("Origin", m_origin_);
    m_axis_ = cfg->GetValue("Axis", m_axis_);
}
std::shared_ptr<simpla::data::DataNode> Axis::Serialize() const {
    auto res = simpla::data::DataNode::New(simpla::data::DataNode::DN_TABLE);
    res->SetValue("Origin", m_origin_);
    res->SetValue("Axis", m_axis_);
    return res;
}
void Axis::Mirror(const point_type &p) { UNIMPLEMENTED; }
void Axis::Mirror(const Axis &a1) { UNIMPLEMENTED; }
void Axis::Rotate(const Axis &a1, Real angle) { UNIMPLEMENTED; }
void Axis::Scale(Real s, int dir) {
    if (dir < 0) {
        m_axis_ *= s;
    } else {
        m_axis_[dir % 3] *= s;
    }
}
void Axis::Translate(const vector_type &v) { m_origin_ += v; }
void Axis::Move(const point_type &p) { m_origin_ = p; }

}  // namespace geometry{
}  // namespace simpla{