//
// Created by salmon on 17-11-19.
//

#ifndef SIMPLA_WIRE_H
#define SIMPLA_WIRE_H
#include <memory>
#include "Edge.h"
namespace simpla {
namespace geometry {
struct Wire : public GeoObject {
    SP_GEO_OBJECT_ABS_HEAD(GeoObject, Wire)
    virtual size_type size() const = 0;
    virtual std::shared_ptr<const Edge> GetEdge(size_type n) = 0;
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_WIRE_H
