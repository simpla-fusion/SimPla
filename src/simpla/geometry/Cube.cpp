//
// Created by salmon on 17-5-31.
//
#include "Cube.h"
namespace simpla {
namespace geometry {
REGISTER_CREATOR(Cube, Cube)

Cube::Cube() = default;
Cube::~Cube() = default;

void Cube::Serialize(const std::shared_ptr<data::DataEntity> &cfg) const {
    base_type::Serialize(cfg);
    auto tdb = std::dynamic_pointer_cast<data::DataTable>(cfg);
    if (tdb != nullptr) { tdb->SetValue("Box", m_bound_box_); }
};
void Cube::Deserialize(const std::shared_ptr<const data::DataEntity> &cfg) {
    base_type::Deserialize(cfg);

    auto tdb = std::dynamic_pointer_cast<const data::DataTable>(cfg);
    if (tdb != nullptr) {
        if (tdb->has("Box")) {
            m_bound_box_ = tdb->GetValue<box_type>("Box");
        } else {
            std::get<0>(m_bound_box_) = tdb->GetValue<nTuple<Real, 3>>("lo", std::get<0>(m_bound_box_));
            std::get<1>(m_bound_box_) = tdb->GetValue<nTuple<Real, 3>>("hi", std::get<1>(m_bound_box_));
        }
    }
}
}
}