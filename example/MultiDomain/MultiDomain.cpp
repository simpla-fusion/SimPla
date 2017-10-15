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
    simpla::Initialize(argc, argv);


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


    TheBegin();

    auto scenario = SimpleTimeIntegrator::New();
    scenario->SetName("MultiDomain");

    scenario->GetAtlas()->SetOrigin({0, 0, 0});
    scenario->GetAtlas()->SetGridWidth({1, 1, 1});
    scenario->GetAtlas()->SetPeriodicDimensions({1, 1, 1});

    scenario->GetAtlas()->NewChart<simpla::geometry::csCartesian>();

    auto center = scenario->NewDomain<SimpleMaxwell>("Center");
    center->SetBoundary(geometry::Cube::New(box_type{{-15, -25, -20}, {15, 25, 20}}));
    center->PostInitialCondition.Connect([=](DomainBase *self, Real time_now) {
        if (auto d = dynamic_cast<SimpleMaxwell *>(self)) {
            d->B = [&](point_type const &x) {

                return point_type{std::sin(2 * PI * x[1] / 50) * std::sin(2 * PI * x[2] / 40),
                                  std::sin(2 * PI * x[0] / 30) * std::sin(2 * PI * x[2] / 40),
                                  std::sin(2 * PI * x[0] / 30) * std::sin(2 * PI * x[1] / 50)};
            };
        }
    });
    //    scenario->NewDomain<SimpleMaxwell>("boundary0")
    //        ->SetBoundary(geometry::Cube::New(box_type{{-20, -25, -20}, {-15, 25, 20}}));
    //    scenario->NewDomain<SimpleMaxwell>("boundary1")
    //        ->SetBoundary(geometry::Cube::New(box_type{{15, -25, -20}, {20, 25, 20}}));
    auto pml = scenario->NewDomain<SimplePML>("PML");
    pml->SetBoundingBox(box_type{{-20, -25, -20}, {20, 25, 20}});
    pml->SetCenterBox(center->GetBoundingBox());

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
    scenario->ConfigureAttribute<size_type>("X10", "CheckPoint", checkpoint_interval);
    scenario->ConfigureAttribute<size_type>("X11", "CheckPoint", checkpoint_interval);
    scenario->ConfigureAttribute<size_type>("X12", "CheckPoint", checkpoint_interval);

    //    VERBOSE << "Scenario: " << *scenario->Serialize();
    scenario->Run();
    //    VERBOSE << "Scenario: " << *scenario->Serialize();

    scenario->TearDown();
    simpla::Finalize();
    TheEnd();
}