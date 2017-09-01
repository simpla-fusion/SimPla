//
// Created by salmon on 17-7-22.
//

#include "Cylindrical.h"
namespace simpla {
namespace geometry {

void Cylindrical::Serialize(std::shared_ptr<simpla::data::DataNode> cfg) const {
    base_type::Serialize(cfg);
    if (cfg != nullptr) {
        cfg->SetValue("Axe0", m_axe0_);
        cfg->SetValue("Axe1", m_axe1_);
        cfg->SetValue("Radius", m_radius_);
    }
};
void Cylindrical::Deserialize(std::shared_ptr<const simpla::data::DataNode> cfg) {
    base_type::Deserialize(cfg);
    if (cfg != nullptr) {
        m_axe0_ = cfg->GetValue("Axe0", m_axe0_);
        m_axe1_ = cfg->GetValue("Axe0", m_axe1_);
        m_radius_ = cfg->GetValue("Radius", m_radius_);
    }
}
}
}