//
// Created by salmon on 17-10-17.
//

#include "Box.h"
#include "GeoObject.h"
#include "simpla/SIMPLA_config.h"

namespace simpla {
namespace geometry {

SP_OBJECT_REGISTER(Box)

Box::Box() = default;
Box::~Box() = default;

std::shared_ptr<data::DataNode> Box::Serialize() const {
    auto cfg = base_type::Serialize();
    if (cfg != nullptr) { cfg->SetValue("Box", m_bound_box_); }
    return cfg;
};
void Box::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
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
}  // namespace geometry
}  // namespace simpla