//
// Created by salmon on 17-11-6.
//

#ifndef SIMPLA_SHAPE_H
#define SIMPLA_SHAPE_H

#include <memory>
#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct Shape : public data::Serializable, public std::enable_shared_from_this<Shape> {
    SP_SERIALIZABLE_HEAD(data::Serializable, Shape)
};

}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_SHAPE_H
