//
// Created by salmon on 17-10-23.
//
#include <simpla/geometry/CutCell.h>
#include <simpla/geometry/GeoEngine.h>
#include <simpla/geometry/Line.h>
#include <simpla/geometry/Polygon.h>
#include <simpla/geometry/Revolution.h>
#include <simpla/geometry/Torus.h>
#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/Log.h>

#include <simpla/geometry/Box.h>
#include <iostream>

namespace sg = simpla::geometry;
using namespace simpla;
int main(int argc, char** argv) {
    logger::set_stdout_level(1000);
    sg::Initialize("OCE");
    GEO_ENGINE->OpenFile("test.stp");
    auto t_surf = sg::Torus::New();
    GEO_ENGINE->Save(t_surf, "Torus");
    GEO_ENGINE->Save(sg::Box::New(), "Box");
    //    auto line = sg::Line::New(point_type{0, 0, 0}, point_type{1, 0, 0});
    //    GEO_ENGINE->Save(line, "Line ");
    //    auto intersect_points = t_surf->GetIntersection(line);
    //    std::cout << *intersect_points->Serialize() << std::endl;
    //    GEO_ENGINE->Save(intersect_points, "Intersection");
    GEO_ENGINE->DumpFile();
    GEO_ENGINE->CloseFile();

    //    auto polygon = sg::Polygon::New();
    //    std::cout << *polygon->Serialize() << std::endl;
    //    std::cout << "Done" << std::endl;
    //    auto revolution = sg::Revolution::New(sg::Axis{point_type{0, 0, 0}, vector_type{0, 0, 1}}, polygon);
    //    std::cout << *revolution->Serialize() << std::endl;
    //    std::cout << "Done" << std::endl;
    //
    //    auto cutcell = sg::CutCell::New();
}
