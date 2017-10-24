//
// Created by salmon on 17-10-24.
//

#include "Revolution.h"
#include "Circle.h"

namespace simpla {
namespace geometry {
Revolution::Revolution() = default;
Revolution::Revolution(Revolution const &other) = default;
Revolution::Revolution(std::shared_ptr<const Surface> const &s, point_type const &origin, vector_type const &axis)
    : SweptBody(s, Circle::New3(origin, s->GetAxis().o, axis)) {}

Revolution::~Revolution() = default;
void Revolution::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> Revolution::Serialize() const {
    auto res = base_type::Serialize();
    return res;
}
point_type Revolution::Value(Real u, Real v, Real w) const   { return m_basis_surface_->Value(u, v); }

}  // namespace geometry{
}  // namespace simpla{