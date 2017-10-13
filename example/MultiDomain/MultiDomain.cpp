//
// Created by salmon on 17-10-11.
//

#include "simpla/SIMPLA_config.h"

#include <simpla/application/SPInit.h>
#include <simpla/geometry/BoxUtilities.h>
#include <simpla/geometry/Cube.h>
#include <simpla/geometry/csCartesian.h>
#include <simpla/mesh/CoRectMesh.h>
#include <simpla/mesh/EBMesh.h>
#include <simpla/mesh/RectMesh.h>
#include <simpla/predefine/engine/SimpleTimeIntegrator.h>
#include <simpla/predefine/physics/Maxwell.h>
#include <simpla/predefine/physics/PML.h>
#include <simpla/scheme/FVM.h>
#include <simpla/utilities/Logo.h>
#include <simpla/utilities/parse_command_line.h>
namespace simpla {
using SimpleMaxwell = domain::Maxwell<engine::Domain<geometry::csCartesian, scheme::FVM, mesh::CoRectMesh>>;
using SimplePML = domain::PML<engine::Domain<geometry::csCartesian, scheme::FVM, mesh::CoRectMesh>>;

}  // namespace simpla {

using namespace simpla;
using namespace simpla::engine;

int main(int argc, char **argv) {
    size_type num_of_step = 10;
    size_type checkpoint_interval = 1;
    simpla::parse_cmd_line(  //
        argc, argv, [&](std::string const &opt, std::string const &value) -> int {
            if (false) {
            } else if (opt == "n") {
                num_of_step = static_cast<size_type>(std::atoi(value.c_str()));
            } else if (opt == "checkpoint") {
                checkpoint_interval = static_cast<size_type>(std::atoi(value.c_str()));
            }
            return CONTINUE;
        });

    simpla::Initialize(argc, argv);
    auto scenario = SimpleTimeIntegrator::New();
    scenario->SetName("MultiDomain");

    box_type center_box{{-15, -25, -20}, {15, 25, 20}};
    box_type bounding_box{{-30, -30, -30}, {20, 30, 30}};

    scenario->GetAtlas()->SetOrigin({0, 0, 0});
    scenario->GetAtlas()->SetGridWidth({1, 1, 1});
    scenario->GetAtlas()->NewChart<simpla::geometry::csCartesian>();

    auto center = scenario->NewDomain<SimpleMaxwell>("Center");
    center->SetBoundary(geometry::Cube::New(center_box));
    center->PostInitialCondition.Connect([=](DomainBase *self, Real time_now) {
        if (auto d = dynamic_cast<SimpleMaxwell *>(self)) {
            d->B = [&](point_type const &x) {
                return point_type{std::cos(2 * PI * x[1] / 60) * std::cos(2 * PI * x[2] / 50),
                                  std::cos(2 * PI * x[0] / 40) * std::cos(2 * PI * x[2] / 50),
                                  std::cos(2 * PI * x[0] / 40) * std::cos(2 * PI * x[1] / 60)};
            };
        }
    });
    auto pml = scenario->NewDomain<SimplePML>("Boundary");
    pml->SetBoundingBox(bounding_box);
    pml->SetCenterBox(center_box);
    scenario->SetTimeEnd(1.0e-8);
    scenario->SetMaxStep(num_of_step);
    scenario->SetUp();

    if (auto atlas = scenario->GetAtlas()) {
        auto box_list = geometry::HaloBoxDecompose(bounding_box, center_box);
        for (auto const &b : box_list) { atlas->NewPatch(b); }
    }

    scenario->ConfigureAttribute<size_type>("E", "CheckPoint", checkpoint_interval);
    scenario->ConfigureAttribute<size_type>("B", "CheckPoint", checkpoint_interval);
    //    scenario->ConfigureAttribute<size_type>("a0", "CheckPoint", 1);
    //    scenario->ConfigureAttribute<size_type>("a1", "CheckPoint", 1);
    //    scenario->ConfigureAttribute<size_type>("a2", "CheckPoint", 1);
    //    scenario->ConfigureAttribute<size_type>("s0", "CheckPoint", 1);
    //    scenario->ConfigureAttribute<size_type>("s1", "CheckPoint", 1);
    //    scenario->ConfigureAttribute<size_type>("s2", "CheckPoint", 1);
    VERBOSE << "Scenario: " << *scenario->Serialize();
    TheStart();
    scenario->Run();
    TheEnd();

    scenario->TearDown();
    simpla::Finalize();
}