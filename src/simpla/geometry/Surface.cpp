//
// Created by salmon on 17-10-19.
//

#include "Surface.h"
#include <simpla/SIMPLA_config.h>
#include "Curve.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

std::shared_ptr<data::DataNode> Surface::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("Axis", m_axis_->Serialize());
    return res;
};
void Surface::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    m_axis_ = Axis::New(cfg->Get("Axis"));
}
// std::shared_ptr<GeoObject> Surface::GetBoundary() const {
//    UNIMPLEMENTED;
//    return nullptr;
//}

}  // namespace geometry
}  // namespace simpla