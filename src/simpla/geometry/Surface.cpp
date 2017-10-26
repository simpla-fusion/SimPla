//
// Created by salmon on 17-10-19.
//

#include "Surface.h"
#include <simpla/SIMPLA_config.h>
#include "Curve.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

std::shared_ptr<data::DataNode> Surface::Serialize() const { return base_type::Serialize(); };
void Surface::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

}  // namespace geometry
}  // namespace simpla