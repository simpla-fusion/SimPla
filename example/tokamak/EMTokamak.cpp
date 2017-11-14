//
// Created by salmon on 16-11-29.
//
#include <simpla/SIMPLA_config.h>
#include <simpla/application/SPInit.h>
#include <simpla/engine/EBDomain.h>
#include <simpla/engine/Engine.h>
#include <simpla/geometry/Box.h>
#include <simpla/geometry/GeoEngine.h>
#include <simpla/geometry/Revolution.h>
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
//#include <simpla/predefine/physics/PICBoris.h>

namespace sg = simpla::geometry;
namespace sp = simpla;
namespace simpla {
typedef engine::Domain<geometry::csCylindrical, scheme::FVM, mesh::CoRectMesh /*, mesh::EBMesh*/> domain_type;

}  // namespace simpla {

using namespace simpla;
using namespace simpla::engine;

int main(int argc, char **argv) {
    sp::Initialize(argc, argv);
    auto scenario = SimpleTimeIntegrator::New();  // SAMRAITimeIntegrator::New();
    scenario->SetName("EAST");
    scenario->db()->SetValue("DumpFileSuffix", "h5");
    scenario->db()->SetValue("CheckPointFilePrefix", "EAST");
    scenario->db()->SetValue("CheckPointFileSuffix", "xmf");

    scenario->GetAtlas()->NewChart<sg::csCylindrical>(point_type{0, 0, 0}, vector_type{0.1, 0.1, PI / 32});
    scenario->GetAtlas()->SetPeriodicDimensions({1, 1, 1});
    scenario->GetAtlas()->SetBoundingBox(box_type{{1.2, -1.4, -PI / 2}, {1.8, 1.4, PI / 2}});
    auto tokamak = sp::Tokamak::New("/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900");
    auto g_boundary = sg::Revolution::New(tokamak->Boundary(), sp::TWOPI);
    auto d_limiter = scenario->NewDomain<domain::Maxwell<domain_type>>(
        "Limiter", sg::Revolution::New(tokamak->Limiter(), sp::TWOPI));
    d_limiter->AddPostInitialCondition([=](simpla::domain::Maxwell<domain_type> *self, Real time_now) {
        self->B = [&](point_type const &x) {
            return point_type{std::cos(2 * PI * x[1] / 60) * std::cos(2 * PI * x[2] / 50),
                              std::cos(2 * PI * x[0] / 40) * std::cos(2 * PI * x[2] / 50),
                              std::cos(2 * PI * x[0] / 40) * std::cos(2 * PI * x[1] / 60)};
        };

    });
    scenario->SetTimeNow(0);
    scenario->SetTimeEnd(1.0e-8);
    scenario->SetMaxStep(50);
    scenario->SetUp();

    scenario->ConfigureAttribute<size_type>("E", "CheckPoint", 1);
    scenario->ConfigureAttribute<size_type>("B", "CheckPoint", 1);

    VERBOSE << "Scenario: " << *scenario->Serialize();

    scenario->Run();

    scenario->TearDown();
    sp::Finalize();
}
