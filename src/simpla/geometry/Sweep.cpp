//
// Created by salmon on 17-10-24.
//

#include "Sweep.h"
namespace simpla {
namespace geometry {

Sweep::Sweep() = default;
Sweep::Sweep(Sweep const &other) = default;
Sweep::Sweep(Axis const &axis) : PrimitiveShape(axis) {}
Sweep::~Sweep() = default;
void Sweep::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataEntry> Sweep::Serialize() const { return base_type::Serialize(); }
Sweep::Sweep(std::shared_ptr<const GeoObject> const &s, std::shared_ptr<const Curve> const &c)
    : m_basis_obj_(s), m_curve_(c){};

}  // namespace geometry{
}  // namespace simpla{