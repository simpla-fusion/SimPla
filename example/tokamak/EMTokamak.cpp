//
// Created by salmon on 16-11-29.
//
#include "simpla/SIMPLA_config.h"

#include <simpla/application/SPInit.h>
#include <simpla/engine/Engine.h>
#include <simpla/geometry/Cube.h>
#include <simpla/geometry/csCartesian.h>
#include <simpla/geometry/csCylindrical.h>
#include <simpla/mesh/CoRectMesh.h>
#include <simpla/mesh/EBMesh.h>
#include <simpla/mesh/RectMesh.h>
#include <simpla/predefine/device/ICRFAntenna.h>
#include <simpla/predefine/device/Tokamak.h>
#include <simpla/predefine/engine/SimpleTimeIntegrator.h>
#include <simpla/predefine/physics/EMFluid.h>
#include <simpla/predefine/physics/Maxwell.h>
#include <simpla/predefine/physics/PICBoris.h>
#include <simpla/scheme/FVM.h>
#include <simpla/third_part/SAMRAITimeIntegrator.h>
#include <simpla/utilities/Logo.h>

namespace simpla {
typedef engine::Domain<geometry::csCartesian, scheme::FVM, mesh::CoRectMesh /*, mesh::EBMesh*/> domain_type;
}  // namespace simpla {

using namespace simpla;
using namespace simpla::engine;

int main(int argc, char **argv) {
    simpla::Initialize(argc, argv);
    //    auto scenario = SAMRAITimeIntegrator::New();
    auto scenario = SimpleTimeIntegrator::New();
    scenario->SetName("EAST");
    scenario->db()->SetValue("DumpFileSuffix", "h5");
    scenario->db()->SetValue("CheckPointFilePrefix", "EAST");
    scenario->db()->SetValue("CheckPointFileSuffix", "xmf");

    scenario->GetAtlas()->SetChart<simpla::geometry::csCartesian>();
    scenario->GetAtlas()->GetChart()->SetScale({1, 1.5, 2});
    scenario->GetAtlas()->GetChart()->SetOrigin({0, 0, 0});
    //    scenario->GetAtlas()->SetBoundingBox(box_type{{1.4, -PI / 4, -1.4}, {2.8, PI / 4, 1.4}});
    //    box_type bounding_box{{0, 0, 0}, {20, 30, 40}};
    box_type bounding_box{{-20, -30, -25}, {20, 30, 25}};

    scenario->GetAtlas()->SetBoundingBox(bounding_box);
    //    auto tokamak = Tokamak::New("/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900");
    //    auto* p = new domain::Maxwell<domain_type>;/*tokamak->Limiter()*/
    scenario->NewDomain<domain::Maxwell<domain_type>>("Limiter");
    scenario->GetDomain("Limiter")->PostInitialCondition.Connect([=](DomainBase *self, Real time_now) {
        if (auto d = dynamic_cast<domain::Maxwell<domain_type> *>(self)) {
            d->B = [&](point_type const &x) {
                return point_type{std::cos(2 * PI * x[1] / 60) * std::cos(2 * PI * x[2] / 50),
                                  std::cos(2 * PI * x[0] / 40) * std::cos(2 * PI * x[2] / 50),
                                  std::cos(2 * PI * x[0] / 40) * std::cos(2 * PI * x[1] / 60)};
            };

            //            d->B[0] = [&](index_type i, index_type j, index_type k) { return static_cast<Real>(i); };
            //            d->B[1] = [&](index_type i, index_type j, index_type k) { return static_cast<Real>(j); };
            //            d->B[2] = [&](index_type i, index_type j, index_type k) { return static_cast<Real>(k); };
            //            d->B = [&](point_type const &x) { return x; };
            //            d->E = [&](point_type const& x) {
            //                return point_type{
            //                    0,                                      // std::cos(0.1 * PI * x[1]) * std::cos(0.1 *
            //                    PI * x[2]),
            //                    -0.1 * PI * std::sin(0.1 * PI * x[0]),  // std::cos(0.1 * PI * x[0]) * std::cos(0.1 *
            //                    PI * x[2]),
            //                    0                                       // * std::cos(0.1 * PI * x[1])
            //                };
            //            };
            //            d->E[1] = GLOBAL_COMM.rank();
            //            d->B[2] = GLOBAL_COMM.rank();
            //            d->J = d->E - curl(d->B);
        }
    });

    //    scenario->SetDomain<Domain<mesh_type, EMFluid>>("Plasma", tokamak->Boundary());
    //    scenario->GetDomain("Plasma")->PreInitialCondition.Connect([=](DomainBase* self, Real time_now) {
    //        if (auto d = dynamic_cast<Domain<mesh_type, EMFluid>*>(self)) { d->ne = tokamak->profile("ne"); }
    //    });
    //    scenario->SetTimeNow(0);
    scenario->SetTimeEnd(1.0e-8);
    scenario->SetMaxStep(50);
    scenario->SetUp();

    scenario->ConfigureAttribute<size_type>("E", "CheckPoint", 1);
    scenario->ConfigureAttribute<size_type>("B", "CheckPoint", 1);


    VERBOSE << "Scenario: " << *scenario->Serialize();
    //    INFORM << "Attributes" << *scenario->GetAttributes() << std::endl;
    //    GLOBAL_COMM.barrier();
    //    if (GLOBAL_COMM.rank() == 0) { std::cout << *scenario->GetAtlas()->GetChart()->Serialize() << std::endl; }
    //    GLOBAL_COMM.barrier();
    //    if (GLOBAL_COMM.rank() == 1) { std::cout << *scenario->GetAtlas()->GetChart()->Serialize() << std::endl; }
    //    GLOBAL_COMM.barrier();
    //    if (GLOBAL_COMM.rank() == 2) { std::cout << *scenario->GetAtlas()->GetChart()->Serialize() << std::endl; }
    //    GLOBAL_COMM.barrier();
    //    if (GLOBAL_COMM.rank() == 3) { std::cout << *scenario->GetAtlas()->GetChart()->Serialize() << std::endl; }
    //    GLOBAL_COMM.barrier();
    TheStart();
    scenario->Run();

    //    std::cout << *scenario->Serialize() << std::endl;

    TheEnd();

    scenario->TearDown();

    simpla::Finalize();
}
