//
// Created by salmon on 17-10-11.
//

#include "simpla/SIMPLA_config.h"

#include <simpla/application/SPInit.h>
#include <simpla/engine/EBDomain.h>
#include <simpla/geometry/Box.h>
#include <simpla/geometry/BoxUtilities.h>
#include <simpla/geometry/csCartesian.h>
#include <simpla/mesh/CoRectMesh.h>
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
namespace sg = simpla::geometry;

int main(int argc, char **argv) {
    simpla::Initialize(argc, argv);
    size_type num_of_step = 10;
    size_type checkpoint_interval = 1;
    simpla::parse_cmd_line(  //
        argc, argv, [&](std::string const &opt, std::string const &value) -> int {
            if (false) {
            } else if (opt == "n") {
                num_of_step = static_cast<size_type>(std::stol(value));
            } else if (opt == "checkpoint") {
                checkpoint_interval = static_cast<size_type>(std::stol(value));
            }
            return CONTINUE;
        });

    auto scenario = SimpleTimeIntegrator::New();
    scenario->SetName("MultiDomain");
    scenario->SetAtlas(Atlas::Create<sg::csCartesian>());

    auto center = scenario->NewDomain<SimpleMaxwell>("Center");
    center->SetBoundary(geometry::Box::New(box_type{{-15, -25, -20}, {15, 25, 20}}));
    center->AddPostInitialCondition([=](auto *self, Real time_now) {
        self->B = [&](point_type const &x) {
            return point_type{std::sin(2 * PI * x[1] / 50) * std::sin(2 * PI * x[2] / 40),
                              std::sin(2 * PI * x[0] / 30) * std::sin(2 * PI * x[2] / 40),
                              std::sin(2 * PI * x[0] / 30) * std::sin(2 * PI * x[1] / 50)};

        };
    });

    auto pml = scenario->NewDomain<SimplePML>("PML");
    CHECK(center->GetBoundary()->GetBoundingBox());
    pml->SetCenterBox(center->GetBoundary()->GetBoundingBox());
    scenario->GetAtlas()->SetBoundingBox(box_type{{-15, -25, -25}, {15, 25, 25}});
    scenario->GetAtlas()->SetPeriodicDimensions({1, 1, 1});

    scenario->SetTimeEnd(1.0e-8);
    scenario->SetMaxStep(num_of_step);
    scenario->SetUp();

    if (auto atlas = scenario->GetAtlas()) {
        auto box_list = geometry::HaloBoxDecompose(
            atlas->GetIndexBox(), atlas->GetChart()->GetIndexBox(center->GetBoundary()->GetBoundingBox()));
        for (auto const &b : box_list) { atlas->AddPatch(b); }
    }

    scenario->ConfigureAttribute<size_type>("E", "CheckPoint", checkpoint_interval);
    scenario->ConfigureAttribute<size_type>("B", "CheckPoint", checkpoint_interval);
    VERBOSE << "Configuration: " << std::endl << *scenario->GetAtlas()->Serialize();

    scenario->Run();

    scenario->Dump();
    scenario->TearDown();
    simpla::Finalize();
}