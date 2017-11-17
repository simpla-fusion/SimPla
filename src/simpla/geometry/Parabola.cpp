//
// Created by salmon on 17-10-22.
//

#include "Parabola.h"

namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Parabola)
Parabola::Parabola() = default;
Parabola::Parabola(Parabola const &) = default;
Parabola::Parabola(Axis const &axis) : base_type(axis){};
Parabola::~Parabola() = default;
void Parabola::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_focal_ = cfg->GetValue<Real>("Focal", m_focal_);
}
std::shared_ptr<simpla::data::DataEntry> Parabola::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("Focal", m_focal_);
    return res;
}

}  // namespace geometry{
}  // namespace simpla{