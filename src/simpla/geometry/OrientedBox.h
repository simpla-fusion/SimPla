//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_ORIENTEDBOX_H
#define SIMPLA_ORIENTEDBOX_H

#include <simpla/utilities/SPDefines.h>

namespace simpla {
namespace geometry {
struct OrientedBox {
    point_type const &Origin() const { return m_origin_; };
    vector_type const *Axies() { return m_axies_; };

    Real measure() const { return dot(m_axies_[0], cross(m_axies_[1], m_axies_[2])); };

   private:
    point_type m_origin_;
    vector_type m_length_;
    vector_type m_axies_[3];
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_ORIENTEDBOX_H
