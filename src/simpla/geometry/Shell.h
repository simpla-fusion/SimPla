//
// Created by salmon on 17-11-6.
//

#ifndef SIMPLA_SHELL_H
#define SIMPLA_SHELL_H
#include <memory>
#include "Face.h"
namespace simpla {
namespace geometry {
struct Shell : public GeoObject {
    SP_GEO_OBJECT_ABS_HEAD(GeoObject, Shell)
    virtual size_type size() const { return 0; };
    virtual std::shared_ptr<const Face> GetFace(size_type n) { return nullptr; };
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_SHELL_H
