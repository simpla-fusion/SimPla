//
// Created by salmon on 17-10-11.
//
#include "simpla/SIMPLA_config.h"

#include <simpla/application/SPInit.h>
#include <simpla/geometry/Cube.h>
#include <simpla/geometry/csCartesian.h>
#include <simpla/mesh/CoRectMesh.h>
#include <simpla/engine/EBDomain.h>
#include <simpla/mesh/RectMesh.h>
#include <simpla/predefine/engine/SimpleTimeIntegrator.h>
#include <simpla/predefine/physics/Maxwell.h>
#include <simpla/scheme/FVM.h>
#include <simpla/utilities/Logo.h>

namespace simpla {
using SimpleMaxwell = domain::Maxwell<engine::Domain<geometry::csCartesian, scheme::FVM, mesh::CoRectMesh>>;
}  // namespace simpla {

using namespace simpla;
using namespace simpla::engine;

int main(int argc, char **argv) {
    simpla::Initialize(argc, argv);
    auto scenario = SimpleTimeIntegrator::New();
    scenario->SetName("SimpleFDTD");
    scenario->GetAtlas()->NewChart<simpla::geometry::csCartesian>();
    scenario->GetAtlas()->GetChart()->SetScale({1, 1.5, 2});
    scenario->GetAtlas()->GetChart()->SetOrigin({0, 0, 0});

    box_type bounding_box{{-20, -30, -25}, {20, 30, 25}};

    scenario->GetAtlas()->SetBoundingBox(bounding_box);

    auto domain = scenario->SetDomain("Center", SimpleMaxwell::New());
    domain->PostInitialCondition.Connect([=](DomainBase *self, Real time_now) {
        if (auto d = dynamic_cast<SimpleMaxwell *>(self)) {
            d->B = [&](point_type const &x) {
                return point_type{std::cos(2 * PI * x[1] / 60) * std::cos(2 * PI * x[2] / 50),
                                  std::cos(2 * PI * x[0] / 40) * std::cos(2 * PI * x[2] / 50),
                                  std::cos(2 * PI * x[0] / 40) * std::cos(2 * PI * x[1] / 60)};
            };
        }
    });

    scenario->SetTimeEnd(1.0e-8);
    scenario->SetMaxStep(50);
    scenario->SetUp();

    scenario->ConfigureAttribute<size_type>("E", "CheckPoint", 1);
    scenario->ConfigureAttribute<size_type>("B", "CheckPoint", 1);

    VERBOSE << "Scenario: " << *scenario->Serialize();

    scenario->Run();

    scenario->TearDown();
    simpla::Finalize();
}
