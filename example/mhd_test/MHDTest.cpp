//
// Created by salmon on 17-5-28.
//

#include <simpla/algebra/all.h>
#include <simpla/engine/all.h>
#include "../../to_delete/CartesianGeometry.h"
#include "../../to_delete/CylindricalCoordinates.h"
#include <simpla/physics/Constants.h>
#include <simpla/predefine/physics/EMFluid.h>
#include <simpla/third_part/SAMRAITimeIntegrator.h>
#include <simpla/utilities/sp_def.h>
#include <iostream>
#include "HyperbolicConservationLaw.h"
namespace simpla {
using namespace engine;
static bool s_RegisterDomain =
    engine::Domain::RegisterCreator<EMFluid<mesh::CartesianCoRectMesh>>(std::string("EMFluidCartesianCoRectMesh")) &&
    engine::Domain::RegisterCreator<HyperbolicConservationLaw<mesh::CartesianCoRectMesh>>(
        std::string("HyperbolicConservationLawCartesianCoRectMesh"));
REGISTER_CREATOR(SAMRAITimeIntegrator)

class MHDTest : public engine::Context {
    SP_OBJECT_HEAD(MHDTest, engine::Context)
   public:
    MHDTest() = default;
    ~MHDTest() override = default;

    SP_DEFAULT_CONSTRUCT(MHDTest);
    DECLARE_REGISTER_NAME("MHDTest");

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const& cfg) override;
};

REGISTER_CREATOR(MHDTest)

std::shared_ptr<data::DataTable> MHDTest::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    res->SetValue<std::string>("Type", "MHDTest");
    res->Set(Context::Serialize());
    return res;
};
void MHDTest::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    if (cfg == nullptr) { return; }
    engine::Context::Initialize();
    engine::Context::Deserialize(cfg);

    index_box_type idx_box{{0, 0, 0}, {1, 1, 1}};
    std::get<1>(idx_box) = cfg->GetValue<nTuple<int, 3>>("Dimensions", nTuple<int, 3>{64, 64, 32});
    GetAtlas().SetIndexBox(idx_box);

    auto d = GetDomain("Main");
    typedef mesh::CartesianCoRectMesh mesh_type;

    d->PostInitialCondition.Connect([&](Domain* self, Real time_now) {

        if (self->check("E", typeid(Field<mesh_type, Real, EDGE, 1>))) {
            auto E = self->GetAttribute<Field<mesh_type, Real, EDGE, 1>>("E");
            E.Clear();

            E = [&](point_type const& x) -> Vec3 {
                Real a = std::sin(TWOPI * x[0]) * std::sin(TWOPI * x[1]);

                CHECK(a);
                return Vec3{0, 0, a};
            };
        }
    });
}
}  // namespace simpla {
