//
// Created by salmon on 17-5-31.
//
#include "Cube.h"
namespace simpla {
namespace geometry {
REGISTER_CREATOR(Cube, Cube)

Cube::Cube() = default;
Cube::~Cube() = default;

void Cube::Serialize(std::shared_ptr<data::DataNode> cfg) const {
    base_type::Serialize(cfg);
    if (cfg != nullptr) { cfg->SetValue("Box", m_bound_box_); }
};
void Cube::Deserialize(std::shared_ptr<const data::DataNode> cfg) {
    base_type::Deserialize(cfg);

    if (cfg != nullptr) {
        if (cfg->Get("Box") != nullptr) {
            m_bound_box_ = cfg->GetValue<box_type>("Box");
        } else {
            std::get<0>(m_bound_box_) = cfg->GetValue<nTuple<Real, 3>>("lo", std::get<0>(m_bound_box_));
            std::get<1>(m_bound_box_) = cfg->GetValue<nTuple<Real, 3>>("hi", std::get<1>(m_bound_box_));
        }
    }
}
}
}