//
// Created by salmon on 17-11-7.
//
#include <simpla/SIMPLA_config.h>
#include <simpla/application/SPInit.h>
#include <simpla/geometry/CutCell.h>
#include <simpla/geometry/GeoEngine.h>
#include <simpla/geometry/GeoObject.h>
#include <simpla/geometry/Revolution.h>
#include <simpla/geometry/csCylindrical.h>
#include <simpla/predefine/device/Tokamak.h>
#include <simpla/utilities/Constants.h>

namespace sp = simpla;
namespace sg = simpla::geometry;

int main(int argc, char **argv) {
    sp::logger::set_stdout_level(1000);

    auto tokamak = sp::Tokamak::New("/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900");
    sg::Initialize("OCE");
    GEO_ENGINE->OpenFile("tokamak.stl");
    auto limiter = sg::Revolution::New(tokamak->Limiter(), sp::PI);
    auto boundary = sg::Revolution::New(tokamak->Boundary(), sp::PI);
    GEO_ENGINE->Save(limiter, "Limiter");
    GEO_ENGINE->Save(boundary, "Boundary");

    auto chart = sg::csCylindrical::New();
    auto cut_cell = sg::CutCell::New(limiter, chart);
    sp::Array<unsigned int> node_tags{{5, 0, 0}, {10, 2, 2}};
    cut_cell->TagCell(&node_tags, nullptr, 0b001);
    GEO_ENGINE->CloseFile();
    sg::Finalize();

    VERBOSE << "DONE";
}