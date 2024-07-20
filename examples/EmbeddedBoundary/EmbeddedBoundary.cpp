//
// Created by salmon on 17-10-16.
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

using namespace simpla;
using namespace simpla::engine;

using EBMaxwell = domain::Maxwell<EBDomain<geometry::csCartesian, scheme::FVM, mesh::CoRectMesh>>;
using SimplePML = domain::PML<EBDomain<geometry::csCartesian, scheme::FVM, mesh::CoRectMesh>>;

int main(int argc, char **argv) {
    simpla::Initialize(argc, argv);

    size_type num_of_step = 10;
    size_type checkpoint_interval = 1;
    simpla::parse_cmd_line(  //
        argc, argv, [&](std::string const &opt, std::string const &value) -> int {
            if (false) {
            } else if (opt == "n") {
                num_of_step = static_cast<size_type>(std::stol(value.c_str()));
            } else if (opt == "checkpoint") {
                checkpoint_interval = static_cast<size_type>(std::stol(value.c_str()));
            }
            return CONTINUE;
        });

    auto scenario = SimpleTimeIntegrator::New();
    scenario->SetName("EmbeddedBoundary");
    scenario->GetAtlas()->NewChart<simpla::geometry::csCartesian>(point_type{0, 0, 0}, point_type{1, 1, 1});

    scenario->GetAtlas()->SetPeriodicDimensions({1, 1, 1});

    auto center = scenario->NewDomain<EBMaxwell>("Center", geometry::Box::New({{-15, -25, -20}, {15, 25, 20}}));

    center->AddEmbeddedDomain<domain::Maxwell>("maxwell", geometry::Box::New({{-5, -5, -5}, {5, 5, 5}}));

    center->PostInitialCondition.Connect([=](DomainBase *self, Real time_now) {
        if (auto d = dynamic_cast<EBMaxwell *>(self)) {
            d->B = [&](point_type const &x) {
                return point_type{std::sin(2 * PI * x[1] / 50) * std::sin(2 * PI * x[2] / 40),
                                  std::sin(2 * PI * x[0] / 30) * std::sin(2 * PI * x[2] / 40),
                                  std::sin(2 * PI * x[0] / 30) * std::sin(2 * PI * x[1] / 50)};
            };
        }
    });

    scenario->SetTimeEnd(1.0e-8);
    scenario->SetMaxStep(num_of_step);
    scenario->SetUp();

    if (auto atlas = scenario->GetAtlas()) {
        scenario->GetAtlas()->Decompose({3, 3, 3});
        auto c_box = center->GetBoundingBox();
        auto box_list = geometry::HaloBoxDecompose(
            atlas->GetIndexBox(),
            std::make_tuple(std::get<1>(atlas->GetChart()->invert_local_coordinates(std::get<0>(c_box))),
                            std::get<1>(atlas->GetChart()->invert_local_coordinates(std::get<1>(c_box)))));
        for (auto const &b : box_list) { atlas->AddPatch(b); }
    }

    scenario->ConfigureAttribute<size_type>("E", "CheckPoint", checkpoint_interval);
    scenario->ConfigureAttribute<size_type>("B", "CheckPoint", checkpoint_interval);
    scenario->ConfigureAttribute<size_type>("node_tag", "CheckPoint", checkpoint_interval);

    VERBOSE << "Scenario: " << *scenario->Serialize();

    scenario->Run();

    scenario->TearDown();
    simpla::Finalize();
}