//
// Created by salmon on 17-11-7.
//
#include <simpla/SIMPLA_config.h>
#include <simpla/application/SPInit.h>
#include <simpla/geometry/GeoEngine.h>
#include <simpla/geometry/GeoObject.h>
#include <simpla/predefine/device/Tokamak.h>

namespace sp = simpla;
namespace sg = simpla::geometry;

int main(int argc, char **argv) {
    sp::Initialize(argc, argv);
    sg::GeoEngine.Initialize("OCE");
    sg::GeoEngine.OpenFile("tokamak.stp");
    auto tokamak = sp::Tokamak::New("/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900");
    sg::GeoEngine.Save("limiter.stp", <#initializer #>);
}