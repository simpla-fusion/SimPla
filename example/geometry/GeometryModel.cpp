//
// Created by salmon on 17-10-23.
//

#include <simpla/geometry/Line.h>
#include <simpla/geometry/Polygon.h>
#include <simpla/geometry/Revolution.h>
#include <simpla/geometry/Toroidal.h>
#include <simpla/utilities/FancyStream.h>
namespace sg = simpla::geometry;
using namespace simpla;
int main(int argc, char** argv) {
    //    auto t_surf = sg::Toroidal::New(1.0);
    //    std::cout << *t_surf->Serialize() << std::endl;
    //    auto line = sg::Line::New(point_type{0, 0, 0}, point_type{1, 0, 0});
    //    std::cout << *line->Serialize() << std::endl;
    //    std::vector<Real> intersect_points;
    //    t_surf->FindIntersection(line, intersect_points, SP_GEO_DEFAULT_TOLERANCE);
    //    std::cout << intersect_points << std::endl;

    auto polygon = sg::Polygon::New();
    std::cout << *polygon->Serialize() << std::endl;
    auto revolution = sg::Revolution::New(polygon, point_type{0, 0, 0}, vector_type{0, 0, 1});
    std::cout << *revolution->Serialize() << std::endl;
    std::cout << "Done" << std::endl;
}
