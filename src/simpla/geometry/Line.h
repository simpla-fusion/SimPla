//
// Created by salmon on 17-11-19.
//

#ifndef SIMPLA_LINE_H
#define SIMPLA_LINE_H
#include <memory>
#include "Edge.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct Line : public Edge {
    SP_GEO_OBJECT_HEAD(Edge, Line)
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_LINE_H
