//
// Created by salmon on 17-10-23.
//

#include <simpla/geometry/Line.h>
#include <simpla/geometry/ToroidalSurface.h>

namespace sg = simpla::geometry;
using namespace simpla;
int main(int argc, char** argv) {
    auto t_surf = sg::ToroidalSurface::New(sg::Axis{}, 1.0, 0.2);
    auto line =
        sg::Line::New(sg::Axis{point_type{0, 0, 0}, point_type{1, 0, 0}, point_type{0, 1, 0}, point_type{0, 0, 1}});
    std::cout << *t_surf->Serialize() << std::endl;
    std::cout << *line->Serialize() << std::endl;
}
