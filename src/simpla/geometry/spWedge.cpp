//
// Created by salmon on 17-11-7.
//

#include "spWedge.h"
#include <simpla/SIMPLA_config.h>
namespace simpla {
namespace geometry {

SP_SHAPE_REGISTER(spWedge)
spWedge::spWedge() = default;
spWedge::spWedge(spWedge const &) = default;
spWedge::~spWedge() = default;
spWedge::spWedge(vector_type const &extents, Real ltx) : m_extents_(extents), m_ltx_(ltx){};

std::shared_ptr<data::DataEntry> spWedge::Serialize() const {
    auto cfg = base_type::Serialize();
    return cfg;
};
void spWedge::Deserialize(std::shared_ptr<data::DataEntry> const &cfg) { base_type::Deserialize(cfg); }

point_type spWedge::xyz(Real u, Real v, Real w) const {
    UNIMPLEMENTED;
    return m_axis_.xyz(u, v, w);
};
point_type Wedge::uvw(Real x, Real y, Real z) const {
    UNIMPLEMENTED;
    return m_axis_.uvw(x, y, z);
};
box_type Wedge::GetBoundingBox() const { return std::make_tuple(m_axis_.o, m_axis_.xyz(m_extents_)); };

}  // namespace geometry
}  // namespace simpla