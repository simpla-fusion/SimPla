//
// Created by salmon on 17-5-31.
//
#include "Cube.h"
namespace simpla {
namespace geometry {
REGISTER_CREATOR(Cube, Cube)

Cube::Cube() = default;
Cube::~Cube() = default;
std::shared_ptr<Cube> Cube::New() { return std::shared_ptr<Cube>(new Cube); }

void Cube::Serialize(data::DataTable &cfg) const {
    base_type::Serialize(cfg);
    cfg.SetValue("Box", m_bound_box_);
};
void Cube::Deserialize(const data::DataTable &cfg) {
    base_type::Deserialize(cfg);
    if (cfg.has("Box")) {
        m_bound_box_ = cfg.GetValue<box_type>("Box");
    } else {
        std::get<0>(m_bound_box_) = cfg.GetValue<nTuple<Real, 3>>("lo", std::get<0>(m_bound_box_));
        std::get<1>(m_bound_box_) = cfg.GetValue<nTuple<Real, 3>>("hi", std::get<1>(m_bound_box_));
    }
}
}
}