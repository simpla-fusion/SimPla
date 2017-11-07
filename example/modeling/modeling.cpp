//
// Created by salmon on 17-11-7.
//
#include <simpla/SIMPLA_config.h>
#include <simpla/application/SPInit.h>
#include <simpla/geometry/GeoEngine.h>
#include <simpla/geometry/GeoObject.h>
#include <simpla/geometry/Revolution.h>
#include <simpla/predefine/device/Tokamak.h>
#include <simpla/utilities/Constants.h>
namespace sp = simpla;
namespace sg = simpla::geometry;

int main(int argc, char **argv) {
    sp::logger::set_stdout_level(1000);
    sg::Initialize("OCE");
    GEO_ENGINE->OpenFile("tokamak.stp");
    auto tokamak = sp::Tokamak::New("/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900");

    GEO_ENGINE->Save(sg::Revolution::New(tokamak->Limiter(), sp::PI), "Limiter");
    GEO_ENGINE->Save(sg::Revolution::New(tokamak->Boundary(), sp::PI), "Boundary");

    GEO_ENGINE->CloseFile();
    sg::Finalize();
}