//
// Created by salmon on 17-10-31.
//

#include "ParametricShape.h"
namespace simpla {
namespace geometry {
box_type Body::GetParameterRange() const { return std::make_tuple(m_uvw_min_, m_uvw_max_); };
box_type Body::GetValueRange() const {}
box_type Body::GetBoundingBox() const { return GetValueRange(); };
}  // namespace geometry{
}  // namespace simpla{impla