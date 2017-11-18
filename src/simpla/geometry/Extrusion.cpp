//
// Created by salmon on 17-10-24.
//

#include "Extrusion.h"
#include "spLine.h"
namespace simpla {
namespace geometry {
Extrusion::Extrusion() = default;
Extrusion::Extrusion(Extrusion const &other) = default;
Extrusion::Extrusion(std::shared_ptr<const Surface> const &s, vector_type const &v) : Sweep(s->GetAxis()) {}
Extrusion::~Extrusion() = default;
void Extrusion::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataEntry> Extrusion::Serialize() const {
    auto res = base_type::Serialize();
    return res;
}

}  // namespace geometry{
}  // namespace simpla{