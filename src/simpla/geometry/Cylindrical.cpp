//
// Created by salmon on 17-7-22.
//

#include "Cylindrical.h"
namespace simpla {
namespace geometry {

void Cylindrical::Serialize(std::shared_ptr<simpla::data::DataEntity> const &cfg) const {
    base_type::Serialize(cfg);
    auto tdb = std::dynamic_pointer_cast<data::DataTable>(cfg);
    if (tdb != nullptr) {
        tdb->SetValue("Axe0", m_axe0_);
        tdb->SetValue("Axe1", m_axe1_);
        tdb->SetValue("Radius", m_radius_);
    }
};
void Cylindrical::Deserialize(std::shared_ptr<const simpla::data::DataEntity> const &cfg) {
    base_type::Deserialize(cfg);
    auto tdb = std::dynamic_pointer_cast<const data::DataTable>(cfg);
    if (tdb != nullptr) {
        m_axe0_ = tdb->GetValue("Axe0", m_axe0_);
        m_axe1_ = tdb->GetValue("Axe0", m_axe1_);
        m_radius_ = tdb->GetValue("Radius", m_radius_);
    }
}
}
}