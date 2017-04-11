//
// Created by salmon on 16-6-29.
//

#include <simpla/algebra/all.h>
#include <simpla/engine/all.h>
#include <simpla/predefine/mesh/CartesianGeometry.h>
//#include <simpla/predefine/mesh/CylindricalGeometry.h>
#include <simpla/data/all.h>

#include "EMFluid.h"
//#include "PML.h"

namespace simpla {

static bool ChartCartesianGeometry_IS_REGISTERED =
    Chart::RegisterCreator<simpla::mesh::CartesianGeometry>("CartesianGeometry");
static bool WorkerCartesianGeometryEMFluid_IS_REGISTERED =
    Worker::RegisterCreator<EMFluid<mesh::CartesianGeometry>>("CartesianGeometry.EMFluid");

void create_scenario(engine::Context *ctx) {
    //    CHECK(ChartCartesianGeometry_IS_REGISTERED);
    //    CHECK(WorkerCartesianGeometryEMFluid_IS_REGISTERED);
    auto domain = std::make_shared<Domain>();
    domain->SetGeoObject(std::shared_ptr<geometry::Cube>(new geometry::Cube{{-0.1, 0.2, 0.0}, {1.2, 1.3, 1.4}}));
    domain->SetChart(std::shared_ptr<Chart>(Chart::Create("CartesianGeometry")));
    domain->SetWorker(std::shared_ptr<Worker>(Worker::Create("CartesianGeometry.EMFluid")));

    std::cout << *domain->Serialize() << std::endl;
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
    //                _impl::_assign(), center_domain->mesh()->range(VERTEX),
    //                std::get<1>(item)["Shape"].as<std::function<Real(point_type const &)>>(),
    //                std::get<1>(item)["Density"].as<std::function<Real(point_type const &)>>());
    //        }
    //    });
    //
    //    typedef PML<mesh_type> pml_type;
    //    auto &pml_domain = ctx->GetDomainView("Vacuum");
    //    auto &pml_worker = pml_domain.AddWorker<PML<mesh_type>>();
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
    //        center_domain->J_src_range = center_mesh->range(mesh::EDGE,
    //        options["Constraints"]["J"]["Box"].as<box_type>());
    //        options["Constraints"]["J"]["Value"].as(&center_domain->J_src_fun);
    //    }
    //
    //    if (options["Constraints"]["E"]) {
    //        center_domain->E_src_range = center_mesh->range(mesh::EDGE,
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
}

}  // namespace simpla