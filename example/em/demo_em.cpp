//
// Created by salmon on 16-6-29.
//

#include <simpla/algebra/all.h>
#include <simpla/application/SpApp.h>
#include <simpla/data/all.h>
#include <simpla/engine/all.h>
#include <simpla/predefine/mesh/CartesianGeometry.h>
#include "simpla/predefine/physics/EMFluid.h"
#include "simpla/predefine/physics/PEC.h"
//#include "PML.h"

namespace simpla {
static bool _PRE_REGISTERED =
    EMFluid<mesh::CartesianGeometry>::is_register && PEC<mesh::CartesianGeometry>::is_register;

struct UseCaseEMFluid : public application::SpApp {
    UseCaseEMFluid() {}
    virtual ~UseCaseEMFluid() {}
    virtual std::shared_ptr<data::DataTable> Serialize() const;
    virtual void Deserialize(std::shared_ptr<data::DataTable>);
    virtual void Run() { m_schedule_->Run(); };

   private:
    std::shared_ptr<engine::Schedule> m_schedule_;
};
SP_REGISITER_APP(UseCaseEMFluid, " EM Fluid ");

std::shared_ptr<data::DataTable> UseCaseEMFluid::Serialize() const {
    return m_schedule_ != nullptr ? m_schedule_->Serialize() : std::make_shared<data::DataTable>();
};

void UseCaseEMFluid::Deserialize(std::shared_ptr<data::DataTable> cfg) {
    m_schedule_ = engine::Schedule::Create("TimeIntegrator");

    auto t = std::dynamic_pointer_cast<engine::TimeIntegrator>(m_schedule_);

    t->SetTime(0);
    t->SetTimeEnd(1);
    t->SetTimeStep(0.1);

    auto domain = t->GetContext()->GetDomain("Center");
    domain->SetGeoObject(std::make_shared<geometry::Cube>(box_type{{-0.1, 0.2, 0.0}, {1.2, 1.3, 1.4}}));
    domain->SetChart("CartesianGeometry");
    domain->SetWorker("EMFluid");
    domain->AddBoundaryCondition("PEC");
};

//    ctx->GetAtlas().db()->SetValue("Origin"_ = {0.0, 0.0, 0.0}, "Dx"_ = {1.0, 1.0, 1.0}, "Dimensions"_ = {0, 0,
//    0});
//
//    ctx->GetAtlas().Decompose(size_tuple{2, 3, 2});
//    ctx->GetAtlas().SetRefineRatio(size_tuple{2, 2, 2});
//    ctx->GetAtlas().AddBlock(index_box_type{{0, 0, 0}, {32, 32, 64}});
//    ctx->GetAtlas().AddBlock(index_box_type{{0, 32, 0}, {32, 64, 64}});
//    ctx->GetAtlas().AddBlock(index_box_type{{32, 0, 0}, {64, 32, 64}});
//    ctx->GetAtlas().AddBlock(index_box_type{{32, 32, 0}, {64, 64, 64}});
//    ctx->GetModel("Center").AddObject("InnerBox", geometry::Cube({{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}));
//    ctx->GetModel("Center").AddObject("OuterBox", geometry::Cube({{-0.1, -0.1, -0.1}, {1.1, 1.1, 1.1}}));
//    ctx->GetModel().db()->SetValue("Center", {"GeoObject"_ = {"InnerBox"}});
//    ctx->GetModel().db()->SetValue("Boundary", {"GeoObject"_ = {"+OuterBox", "-InnerBox"}});
//    ctx->db()->SetValue("Domain/Center", {"Mesh"_ = "CartesianGeometry", "Worker"_ = {{"name"_ = "EMFluid"}}});
//    ctx->db()->SetValue("Domain/Boundary", {"Mesh"_ = "CartesianGeometry", "Task"_ = {{"name"_ = "PML"}}});
//    options.GetTable("Particles").foreach ([&](auto const &item) {
//        auto sp = center_worker.AddSpecies(std::get<0>(item).template as<std::string>(),
//                                           std::get<1>(item)["Mass"].template as<Real>(1.0),
//                                           std::get<1>(item)["Charge"].template as<Real>(1.0));
//        sp->rho->Clear();
//        sp->J->Clear();
//        if (std::get<1>(item).has("Density")) {
//            sp->rho.apply_function_with_define_domain(
//                _impl::_assign(), center_domain->mesh()->GetRange(VERTEX),
//                std::get<1>(item)["Shape"].as<std::function<Real(point_type const &)>>(),
//                std::get<1>(item)["Density"].as<std::function<Real(point_type const &)>>());
//        }
//    });
//
//    typedef PML<mesh_type> pml_type;
//    auto &pml_domain = ctx->GetDomainView("Vacuum");
//    auto &pml_worker = pml_domain.SetWorker<PML<mesh_type>>();
//    pml_worker.SetCenterDomain(in_box);
//    typedef std::function<vector_type(point_type const &)> field_function_type;
//
//    if (options.has("InitValue")) {
//        if (options.has("InitValue.B0")) {
//            center_domain->B0.assign_function(options["InitValue"]["B0"]["Value"].as<field_function_type>());
//        }
//
//        if (options["InitValue"]["B1"]) {
//            center_domain->B.assign_function(options["InitValue"]["B1"]["Value"].as<field_function_type>());
//        }
//
//        if (options["InitValue"]["E1"]) {
//            center_domain->E.assign_function(options["InitValue"]["E1"]["Value"].as<field_function_type>());
//        }
//    }
//
//    if (options["Constraints"]["J"]) {
//        center_domain->J_src_range = center_mesh->GetRange(mesh::EDGE,
//        options["Constraints"]["J"]["Box"].as<box_type>());
//        options["Constraints"]["J"]["Value"].as(&center_domain->J_src_fun);
//    }
//
//    if (options["Constraints"]["E"]) {
//        center_domain->E_src_range = center_mesh->GetRange(mesh::EDGE,
//        options["Constraints"]["E"]["Box"].as<box_type>());
//        options["Constraints"]["E"]["Value"].as(&center_domain->E_src_fun);
//    }
//
//    if (options["Constraints"]["PEC"]) {
//        mesh::Modeler model(center_mesh.get());
//        std::function<Real(point_type const &)> shape_fun;
//        options["Constraints"]["PEC"]["Shape"].as(&shape_fun);
//
//        model.add(options["Constraints"]["PEC"]["Box"].as<box_type>(), shape_fun);
//
//        center_domain->face_boundary = model.surface(FACE);
//        center_domain->edge_boundary = model.surface(EDGE);
//
//        options["Constraints"]["J"]["Value"].as(&center_domain->J_src_fun);
//    }

}  // namespace simpla