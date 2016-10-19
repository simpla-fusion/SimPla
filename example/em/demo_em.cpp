//
// Created by salmon on 16-6-29.
//


#include "simulation/Context.h"
#include "manifold/pre_define/PreDefine.h"
#include "../../scenario/problem_domain/EMFluid.h"
#include "../../scenario/problem_domain/PML.h"

namespace simpla
{

void create_scenario(simulation::Context *ctx, toolbox::ConfigParser const &options)
{
    typedef manifold::CartesianManifold mesh_type;

    auto center_domain = ctx->add_domain<EMFluid<mesh_type >>();

    center_domain->mesh()->name("Center");
    center_domain->mesh()->dimensions(options["Mesh"]["Dimensions"].template as<index_tuple>(size_tuple{20, 20, 1}));
    center_domain->mesh()->ghost_width(options["Mesh"]["GhostWidth"].template as<index_tuple>(size_tuple{2, 2, 2}));
    center_domain->mesh()->box(options["Mesh"]["MeshBase"].template as<box_type>(box_type{{{0, 0, 0}, {1, 1, 1}}}));
    center_domain->deploy();

    auto center_mesh = center_domain->mesh();

    if (options["PML"])
    {
        typedef PML<mesh_type> pml_type;

        index_type w = options["PML"]["Width"].as<index_type>(5);
        size_tuple dims = center_mesh->dimensions();
        size_tuple gw = center_mesh->ghost_width();

        if (dims[0] > 1 && gw[0] > 0)
        {
            auto pml0 = ctx->add_domain<pml_type>(center_mesh->clone("PML_0"));
            pml0->mesh()->shift(-w, -w, -w);
            pml0->mesh()->reshape(w, dims[1] + 2 * w, dims[2] + 2 * w);
            pml0->setup_center_domain(center_mesh->box());
            pml0->deploy();

            auto pml1 = ctx->add_domain<pml_type>(center_mesh->clone("PML_1"));
            pml0->mesh()->shift(dims[0], -w, -w);
            pml0->mesh()->reshape(w, dims[1] + 2 * w, dims[2] + 2 * w);
            pml0->setup_center_domain(center_mesh->box());
            pml0->deploy();
        }

        if (dims[1] > 1 && gw[1] > 0)
        {
            auto pml2 = ctx->add_domain<pml_type>(center_mesh->clone("PML_2"));
            pml2->mesh()->shift(0, -w, -w);
            pml2->mesh()->reshape(dims[0], w, dims[2] + 2 * w);
            pml2->setup_center_domain(center_mesh->box());
            pml2->deploy();

            auto pml3 = ctx->add_domain<pml_type>(center_mesh->clone("PML_3"));
            pml3->mesh()->shift(0, dims[1], -w);
            pml3->mesh()->reshape(dims[0], w, dims[2] + 2 * w);
            pml3->setup_center_domain(center_mesh->box());
            pml3->deploy();

        }
        if (dims[2] > 1 && gw[1] > 0)
        {
            auto pml4 = ctx->add_domain<pml_type>(center_mesh->clone("PML_4"));
            pml4->mesh()->shift(0, 0, -w);
            pml4->mesh()->reshape(dims[0], dims[1], w);
            pml4->setup_center_domain(center_mesh->box());
            pml4->deploy();

            auto pml5 = ctx->add_domain<pml_type>(center_mesh->clone("PML_5"));
            pml5->mesh()->shift(0, 0, dims[2]);
            pml5->mesh()->reshape(dims[0], dims[1], w);
            pml5->setup_center_domain(center_mesh->box());
            pml5->deploy();

        }
    }

    ctx->deploy();

    for (auto const &item:options["Particles"])
    {
        auto sp = center_domain->add_particle(std::get<0>(item).template as<std::string>(),
                                              std::get<1>(item)["Mass"].template as<Real>(1.0),
                                              std::get<1>(item)["Charge"].template as<Real>(1.0));
        sp->rho.clear();
        sp->J.clear();
        if (std::get<1>(item)["Density"])
        {
            sp->rho.apply_function_with_define_domain(
                    _impl::_assign(), center_domain->mesh()->range(VERTEX),
                    std::get<1>(item)["Shape"].as<std::function<Real(point_type const &)> >(),
                    std::get<1>(item)["Density"].as<std::function<Real(point_type const &)>>()
            );
        }
    }

    typedef std::function<vector_type(point_type const &)> field_function_type;

    if (options["InitValue"])
    {
        if (options["InitValue"]["B0"])
        {
            center_domain->B0.assign_function(options["InitValue"]["B0"]["Value"].as<field_function_type>());
        }

        if (options["InitValue"]["B1"])
        {
            center_domain->B.assign_function(options["InitValue"]["B1"]["Value"].as<field_function_type>());
        }

        if (options["InitValue"]["E1"])
        {
            center_domain->E.assign_function(options["InitValue"]["E1"]["Value"].as<field_function_type>());
        }
    }


    if (options["Constraints"]["J"])
    {
        center_domain->J_src_range = center_mesh->range(mesh::EDGE, options["Constraints"]["J"]["Box"].as<box_type>());
        options["Constraints"]["J"]["Value"].as(&center_domain->J_src_fun);
    }

    if (options["Constraints"]["E"])
    {
        center_domain->E_src_range = center_mesh->range(mesh::EDGE, options["Constraints"]["E"]["Box"].as<box_type>());
        options["Constraints"]["E"]["Value"].as(&center_domain->E_src_fun);
    }


    if (options["Constraints"]["PEC"])
    {
        mesh::Model model(center_mesh.get());
        std::function<Real(point_type const &)> shape_fun;
        options["Constraints"]["PEC"]["Shape"].as(&shape_fun);

        model.add(options["Constraints"]["PEC"]["Box"].as<box_type>(), shape_fun);

        center_domain->face_boundary = model.surface(FACE);
        center_domain->edge_boundary = model.surface(EDGE);

        options["Constraints"]["J"]["Value"].as(&center_domain->J_src_fun);
    }
}

}//namespace simpla