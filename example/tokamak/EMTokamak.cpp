//
// Created by salmon on 16-11-29.
//
#include <simpla/SIMPLA_config.h>
#include <simpla/application/SPInit.h>
#include <simpla/engine/EBDomain.h>
#include <simpla/engine/Engine.h>
#include <simpla/geometry/Box.h>
#include <simpla/geometry/Edge.h>
#include <simpla/geometry/GeoEngine.h>
#include <simpla/geometry/Sweeping.h>
#include <simpla/geometry/csCartesian.h>
#include <simpla/geometry/csCylindrical.h>
#include <simpla/mesh/CoRectMesh.h>
#include <simpla/mesh/RectMesh.h>
#include <simpla/predefine/device/ICRFAntenna.h>
#include <simpla/predefine/device/Tokamak.h>
#include <simpla/predefine/engine/SimpleTimeIntegrator.h>
#include <simpla/predefine/physics/EMFluid.h>
#include <simpla/predefine/physics/Maxwell.h>
#include <simpla/scheme/FVM.h>
#include <simpla/utilities/Logo.h>
namespace sg = simpla::geometry;
namespace sp = simpla;
namespace simpla {
typedef engine::Domain<geometry::csCylindrical, scheme::FVM, mesh::CoRectMesh /*, mesh::EBMesh*/> domain_type;

}  // namespace simpla {

using namespace simpla;
using namespace simpla::engine;

int main(int argc, char** argv) {
    sp::Initialize(argc, argv);
    auto scenario = SimpleTimeIntegrator::New();  // SAMRAITimeIntegrator::New();
    scenario->SetProperty<std::string>("Name", "EAST");
    scenario->SetProperty<std::string>("DumpFileSuffix", "h5");
    scenario->SetProperty<std::string>("CheckPointFilePrefix", "EAST");
    scenario->SetProperty<std::string>("CheckPointFileSuffix", "xdmf");

    scenario->GetAtlas()->NewChart<sg::csCylindrical>(point_type{0, 0, 0}, vector_type{0.1, 0.1, PI / 32});
    scenario->GetAtlas()->SetPeriodicDimensions({1, 1, 1});
    scenario->GetAtlas()->SetBoundingBox(box_type{{1.2, -1.4, -PI / 2}, {1.8, 1.4, PI / 2}});
    auto tokamak = sp::Tokamak::New("/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900");
    //    auto g_boundary = sg::MakeRevolution(tokamak->Boundary(), sg::Axis{}, TWOPI);
    auto d_limiter = scenario->NewDomain<domain::Maxwell<domain_type>>(
        "Limiter", sg::MakeRevolution(tokamak->Limiter(), vector_type{1, 1, 0}));
    d_limiter->AddPostInitialCondition([=](auto* self, Real time_now) {
        self->B = [&](point_type const& x) {
            return point_type{std::cos(2 * PI * x[1] / 60) * std::cos(2 * PI * x[2] / 50),
                              std::cos(2 * PI * x[0] / 40) * std::cos(2 * PI * x[2] / 50),
                              std::cos(2 * PI * x[0] / 40) * std::cos(2 * PI * x[1] / 60)};
        };

    });
    scenario->SetTimeNow(0);
    scenario->SetTimeEnd(1.0e-8);
    scenario->SetMaxStep(5);
    scenario->SetUp();
    scenario->GetAtlas()->AddPatch(scenario->GetAtlas()->GetBoundingBox());
    scenario->ConfigureAttribute<size_type>("E", "CheckPoint", 1);
    scenario->ConfigureAttribute<size_type>("B", "CheckPoint", 1);
    VERBOSE << "Scenario: " << *scenario->Serialize();
    scenario->Run();
    scenario->Dump();
    scenario->TearDown();
    sp::Finalize();
}
