//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_AXIS_H
#define SIMPLA_AXIS_H

#include <simpla/data/DataNode.h>
namespace simpla {
namespace geometry {
struct Axis {
    Axis() = default;
    Axis(Axis const &) = default;
    ~Axis() = default;
    Axis(point_type origin, point_type x_axis, point_type y_axis, point_type z_axis)
        : o(std::move(origin)), x(std::move(x_axis)), y(std::move(y_axis)), z(std::move(z_axis)) {}

    void Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg);
    std::shared_ptr<simpla::data::DataNode> Serialize() const;

    void O(point_type const &p) { o = p; }
    void X(point_type const &p) { x = p; }
    void Y(point_type const &p) { y = p; }
    void Z(point_type const &p) { y = p; }

    point_type const &O() const { return o; }
    point_type const &X() const { return x; }
    point_type const &Y() const { return y; }
    point_type const &Z() const { return z; }

    point_type o{0, 0, 0};
    vector_type x{1, 0, 0};
    vector_type y{0, 1, 0};
    vector_type z{0, 0, 1};
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_AXIS_H
