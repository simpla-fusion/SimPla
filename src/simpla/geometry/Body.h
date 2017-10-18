//
// Created by salmon on 17-10-18.
//

#ifndef SIMPLA_BODY_H
#define SIMPLA_BODY_H

#include <memory>
#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct Shell;
struct Body : public GeoObject {
    SP_OBJECT_HEAD(Body, GeoObject)

    virtual std::shared_ptr<Shell> GetShell() const;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BODY_H
