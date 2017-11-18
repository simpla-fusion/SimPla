//
// Created by salmon on 17-10-22.
//

#include "spParabola.h"

namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(spParabola)
spParabola::spParabola() = default;
spParabola::spParabola(spParabola const &) = default;
spParabola::~spParabola() = default;
void spParabola::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_focal_ = cfg->GetValue<Real>("Focal", m_focal_);
}
std::shared_ptr<simpla::data::DataEntry> spParabola::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("Focal", m_focal_);
    return res;
}

}  // namespace geometry{
}  // namespace simpla{