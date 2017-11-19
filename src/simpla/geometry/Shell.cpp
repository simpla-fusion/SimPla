//
// Created by salmon on 17-11-6.
//

#include "Shell.h"
#include "GeoEntity.h"
namespace simpla {
namespace geometry {
Shell::Shell(Shell const &) = default;
Shell::Shell(Axis const &axis) : GeoObject(axis){};

}  // namespace geometry{
}  // namespace simpla{