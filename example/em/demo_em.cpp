//
// Created by salmon on 16-6-29.
//

#include <simpla/engine/Atlas.h>
#include <simpla/engine/DomainView.h>
#include <simpla/engine/Manager.h>
#include <simpla/model/geometry/Cube.h>
#include <simpla/predefine/CartesianGeometry.h>
#include "../../scenario/problem_domain/EMFluid.h"
#include "../../scenario/problem_domain/PML.h"

namespace simpla {

void create_scenario(engine::Manager *ctx) {
    typedef mesh::CartesianGeometry mesh_type;

    engine::Atlas &atlas = ctx->GetAtlas();

    atlas.SetOrigin(point_type({0, 0, 0}));
    atlas.SetDx(point_type({1, 1, 1}));

    model::Model &model = ctx->GetModel();
    geometry::Cube in_box(ctx->db<box_type>("Model/Geometry/InnerBox", box_type{{0, 0, 0}, {1, 1, 1}}));
    geometry::Cube out_box{ctx->db<box_type>("Model/Geometry/OuterBox", box_type{{-0.1, -0.1, -0.1}, {1.1, 1.1, 1.1}})};
    model.AddObject("Plasma", in_box);
    model.AddObject("Vacuum", out_box - in_box);
    auto &center_domain = ctx->GetDomainView("Plasma");
    mesh_type &center_mesh = center_domain.SetMesh<mesh_type>();
    auto &center_worker = center_domain.AddWorker<EMFluid<mesh_type>>();

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

    typedef PML<mesh_type> pml_type;
    auto &pml_domain = ctx->GetDomainView("Vacuum");
    auto &pml_worker = pml_domain.AddWorker<PML<mesh_type>>();
    pml_worker.SetCenterDomain(in_box);

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