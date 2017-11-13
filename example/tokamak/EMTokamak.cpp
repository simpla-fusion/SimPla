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
//#include <simpla/predefine/physics/PICBoris.h>
#include <simpla/scheme/FVM.h>
#include <simpla/utilities/Logo.h>

namespace sg = simpla::geometry;
namespace sp = simpla;
namespace simpla {
typedef engine::Domain<geometry::csCartesian, scheme::FVM, mesh::CoRectMesh /*, mesh::EBMesh*/> domain_type;
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
    scenario->GetAtlas()->NewChart<sg::csCartesian>(point_type{0, 0, 0}, point_type{1, 1.5, 2});
    scenario->GetAtlas()->SetPeriodicDimensions({1, 1, 1});

    auto tokamak = sp::Tokamak::New("/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900");

    auto g_boundary = sg::Revolution::New(tokamak->Boundary(), sp::TWOPI);

    auto limiter = scenario->NewDomain<domain::Maxwell<domain_type>>(
        "Limiter", sg::Revolution::New(tokamak->Limiter(), sp::TWOPI));
    limiter->PostInitialCondition.Connect([=](DomainBase *self, Real time_now) {
        if (auto d = dynamic_cast<domain::Maxwell<domain_type> *>(self)) {
            d->B = [&](point_type const &x) {
                return point_type{std::cos(2 * PI * x[1] / 60) * std::cos(2 * PI * x[2] / 50),
                                  std::cos(2 * PI * x[0] / 40) * std::cos(2 * PI * x[2] / 50),
                                  std::cos(2 * PI * x[0] / 40) * std::cos(2 * PI * x[1] / 60)};
            };
        }
    });

    //    scenario->SetDomain<Domain<mesh_type, EMFluid>>("Plasma", tokamak->Boundary());
    //    scenario->GetDomain("Plasma")->PreInitialCondition.Connect([=](DomainBase* self, Real time_now) {
    //        if (auto d = dynamic_cast<Domain<mesh_type, EMFluid>*>(self)) { d->ne = tokamak->profile("ne"); }
    //    });
    scenario->SetTimeNow(0);
    scenario->SetTimeEnd(1.0e-8);
    scenario->SetMaxStep(50);
    scenario->SetUp();

    scenario->ConfigureAttribute<size_type>("E", "CheckPoint", 1);
    scenario->ConfigureAttribute<size_type>("B", "CheckPoint", 1);

    VERBOSE << "Scenario: " << *scenario->Serialize();

    scenario->Run();
    //    std::cout << *scenario->Serialize() << std::endl;

    scenario->TearDown();
    sp::Finalize();
}
