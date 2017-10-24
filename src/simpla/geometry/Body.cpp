//
// Created by salmon on 17-10-18.
//
#include "Body.h"
#include "Curve.h"
namespace simpla {
namespace geometry {

std::shared_ptr<data::DataNode> Body::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("Axis", m_axis_.Serialize());
    return res;
};
void Body::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_axis_.Deserialize(cfg->Get("Axis"));
}

/**
* @return
*  <= 0 no overlap
*  == 1 partial overlap
*  >  1 all inside
*/
int Body::CheckOverlap(box_type const &, Real tolerance) const { return 0; }
/**
 *
 * @return <0 first point is outgoing
 *         >0 first point is incoming
 */
int Body::FindIntersection(std::shared_ptr<const Curve> const &, std::vector<Real> &, Real tolerance) const {}
// std::shared_ptr<GeoObject> Surface::GetBoundary() const {
//    UNIMPLEMENTED;
//    return nullptr;
//}

}  // namespace geometry
}  // namespace simpla