//
// Created by salmon on 17-10-23.
//

#include <simpla/geometry/CutCell.h>
#include <simpla/geometry/GeoEngine.h>
#include <simpla/geometry/Line.h>
#include <simpla/geometry/Polygon.h>
#include <simpla/geometry/Revolution.h>
#include <simpla/geometry/Toroidal.h>
#include <simpla/utilities/FancyStream.h>

namespace sg = simpla::geometry;
using namespace simpla;
int main(int argc, char** argv) {
    logger::set_stdout_level(1000);
    sg::GeoEngine::Initialize("OCE");
    CHECK(sg::GeoEngine::RegisterName_s());
    auto t_surf = sg::Toroidal::New();
    std::cout << *t_surf->Serialize() << std::endl;
    auto line = sg::Line::New(point_type{0, 0, 0}, point_type{1, 0, 0});
    std::cout << *line->Serialize() << std::endl;
    auto intersect_points = t_surf->GetIntersection(line);
    std::cout << intersect_points << std::endl;

    auto polygon = sg::Polygon::New();
    std::cout << *polygon->Serialize() << std::endl;
    std::cout << "Done" << std::endl;
    auto revolution = sg::Revolution::New(sg::Axis{point_type{0, 0, 0}, vector_type{0, 0, 1}}, polygon);
    std::cout << *revolution->Serialize() << std::endl;
    std::cout << "Done" << std::endl;

    auto cutcell = sg::CutCell::New();
}
