//
// Created by salmon on 16-6-29.
//


#include "../../src/simulation/Context.h"
#include "../../src/manifold/pre_define/PreDefine.h"
#include "../../src/mesh/ModelSelect.h"

#include "../../scenario/problem_domain/EMFluid.h"
#include "../../scenario/problem_domain/PML.h"


namespace simpla
{

void create_scenario(simulation::Context *ctx, toolbox::ConfigParser const &options)
{
    typedef simpla::manifold::CartesianManifold mesh_type;

    auto center_mesh = ctx->add_mesh<mesh_type>();
    center_mesh->name("Center");
    center_mesh->dimensions(options["Mesh"]["Dimensions"].template as<index_tuple>(index_tuple{20, 20, 1}));
    center_mesh->ghost_width(options["Mesh"]["GhostWidth"].template as<index_tuple>(index_tuple{2, 2, 2}));
    center_mesh->box(options["Mesh"]["Block"].template as<box_type>(box_type{{0, 0, 0},
                                                                             {1, 1, 1}}));
    center_mesh->deploy();

    auto center_domain = ctx->add_domain_to<EMFluid<mesh_type >>(center_mesh->id());

    for (auto const &item:options["Particles"])
    {
        auto sp = center_domain->add_particle(std::get<0>(item).template as<std::string>(),
                                              std::get<1>(item)["Mass"].template as<Real>(1.0),
                                              std::get<1>(item)["Charge"].template as<Real>(1.0));
        sp->rho.clear();
        sp->J.clear();
        if (std::get<1>(item)["Density"])
        {
            std::function<Real(point_type const &)> g_obj;
            std::get<1>(item)["Shape"].as(&g_obj);

            std::function<Real(point_type const &)> density;
            std::get<1>(item)["Density"].as(&density);

            center_mesh->foreach(VERTEX,
                                 [&](MeshEntityId const &s)
                                 {
                                     auto x = center_mesh->point(s);
                                     if (g_obj(x) <= 0) { sp->rho[s] = density(x); }
                                 });
        }
    }

    center_domain->deploy();


    if (options["InitValue"])
    {
        if (options["InitValue"]["B0"])
        {
            std::function<vector_type(point_type const &)> fun;
            options["InitValue"]["B0"]["Value"].as(&fun);
            center_mesh->foreach(FACE,
                                 [&](mesh::MeshEntityId const &s)
                                 {
                                     center_domain->B0[s] = fun(center_mesh->point(s))[MeshEntityIdCoder::sub_index(
                                             s)];
                                 });
        }

        if (options["InitValue"]["B1"])
        {
            std::function<vector_type(point_type const &)> fun;
            options["InitValue"]["B1"]["Value"].as(&fun);
            center_mesh->foreach(FACE,
                                 [&](mesh::MeshEntityId const &s)
                                 {
                                     center_domain->B[s] = fun(center_mesh->point(s))[MeshEntityIdCoder::sub_index(s)];
                                 });
        }

        if (options["InitValue"]["E1"])
        {
            std::function<vector_type(point_type const &)> fun;
            options["InitValue"]["E1"]["Value"].as(&fun);
            center_mesh->foreach(EDGE,
                                 [&](mesh::MeshEntityId const &s)
                                 {
                                     center_domain->E[s] = fun(center_mesh->point(s))[MeshEntityIdCoder::sub_index(s)];
                                 });
        }
    }


    if (options["Constraints"]["J"])
    {

        center_domain->J_src_range = select(center_mesh, mesh::EDGE,
                                            options["Constraints"]["J"]["Block"].as<box_type>());

        options["Constraints"]["J"]["Value"].as(&center_domain->J_src_fun);
    }

    if (options["Constraints"]["E"])
    {

        center_domain->E_src_range = select(center_mesh, mesh::EDGE,
                                            options["Constraints"]["E"]["Block"].as<box_type>());

        options["Constraints"]["E"]["Value"].as(&center_domain->E_src_fun);
    }


    if (options["Constraints"]["PEC"])
    {
        mesh::Model model(center_mesh.get());

        std::function<Real(point_type const &)> shape_fun;

        options["Constraints"]["PEC"]["Shape"].as(&shape_fun);

        model.add(options["Constraints"]["PEC"]["Block"].as<box_type>(), shape_fun);

        center_domain->face_boundary = model.surface(FACE);
        center_domain->edge_boundary = model.surface(EDGE);

        options["Constraints"]["J"]["Value"].as(&center_domain->J_src_fun);
    }

    if (options["PML"])
    {
        typedef PML<mesh_type> pml_type;
        std::shared_ptr<mesh_type> pml_mesh[6];
        index_type w = options["PML"]["Width"].as<index_type>(5);
        index_tuple dims;
        dims = center_mesh->dimensions();
        size_tuple gw = center_mesh->ghost_width();

        pml_mesh[0] = std::dynamic_pointer_cast<mesh_type>(center_mesh->clone("PML_0"));
        pml_mesh[0]->shift(index_tuple{-w, -w, -w});
        pml_mesh[0]->stretch(index_tuple{w, dims[1] + 2 * w, dims[2] + 2 * w});
        pml_mesh[0]->deploy();
        ctx->atlas().add_block(pml_mesh[0]);
        ctx->atlas().add_connection(center_mesh, pml_mesh[0]);

        ctx->add_domain_as<pml_type>(pml_mesh[0])->setup_center_domain(center_mesh->box()).deploy();


        pml_mesh[1] = std::dynamic_pointer_cast<mesh_type>(center_mesh->clone("PML_1"));
        pml_mesh[1]->shift(index_tuple{dims[0], -w, -w});
        pml_mesh[1]->stretch(index_tuple{w, dims[1] + 2 * w, dims[2] + 2 * w});
        pml_mesh[1]->deploy();
        ctx->atlas().add_block(pml_mesh[1]);
        ctx->atlas().add_connection(center_mesh, pml_mesh[1]);
        ctx->add_domain_as<pml_type>(pml_mesh[1])->setup_center_domain(center_mesh->box()).deploy();


        if (dims[1] > 1 && gw[1] > 0)
        {
            pml_mesh[2] = std::dynamic_pointer_cast<mesh_type>(center_mesh->clone("PML_2"));
            pml_mesh[2]->shift(index_tuple{0, -w, -w});
            pml_mesh[2]->stretch(index_tuple{dims[0], w, dims[2] + 2 * w});
            pml_mesh[2]->deploy();
            ctx->atlas().add_block(pml_mesh[2]);
            ctx->atlas().add_connection(pml_mesh[2], center_mesh);
            ctx->atlas().add_connection(pml_mesh[2], pml_mesh[0]);
            ctx->atlas().add_connection(pml_mesh[2], pml_mesh[1]);
            ctx->add_domain_as<pml_type>(pml_mesh[2])->setup_center_domain(center_mesh->box()).deploy();


            pml_mesh[3] = std::dynamic_pointer_cast<mesh_type>(center_mesh->clone("PML_3"));
            pml_mesh[3]->shift(index_tuple{0, dims[1], -w});
            pml_mesh[3]->stretch(index_tuple{dims[0], w, dims[2] + 2 * w});
            pml_mesh[3]->deploy();
            ctx->atlas().add_block(pml_mesh[3]);
            ctx->atlas().add_connection(pml_mesh[3], center_mesh);
            ctx->atlas().add_connection(pml_mesh[3], pml_mesh[0]);
            ctx->atlas().add_connection(pml_mesh[3], pml_mesh[1]);
            ctx->add_domain_as<pml_type>(pml_mesh[3])->setup_center_domain(center_mesh->box()).deploy();

        }
        if (dims[2] > 1 && gw[1] > 0)
        {
            pml_mesh[4] = std::dynamic_pointer_cast<mesh_type>(center_mesh->clone("PML_4"));
            pml_mesh[4]->shift(index_tuple{0, 0, -w});
            pml_mesh[4]->stretch(index_tuple{dims[0], dims[1], w});
            pml_mesh[4]->deploy();
            ctx->atlas().add_block(pml_mesh[4]);
            ctx->atlas().add_connection(pml_mesh[4], center_mesh);
            ctx->atlas().add_connection(pml_mesh[4], pml_mesh[0]);
            ctx->atlas().add_connection(pml_mesh[4], pml_mesh[1]);
            ctx->atlas().add_connection(pml_mesh[4], pml_mesh[2]);
            ctx->atlas().add_connection(pml_mesh[4], pml_mesh[3]);
            ctx->add_domain_as<pml_type>(pml_mesh[4])->setup_center_domain(center_mesh->box()).deploy();

            pml_mesh[5] = std::dynamic_pointer_cast<mesh_type>(center_mesh->clone("PML_5"));
            pml_mesh[5]->shift(index_tuple{0, 0, dims[2]});
            pml_mesh[5]->stretch(index_tuple{dims[0], dims[1], w});
            pml_mesh[5]->deploy();
            ctx->atlas().add_block(pml_mesh[5]);
            ctx->atlas().add_connection(pml_mesh[5], center_mesh);
            ctx->atlas().add_connection(pml_mesh[5], pml_mesh[0]);
            ctx->atlas().add_connection(pml_mesh[5], pml_mesh[1]);
            ctx->atlas().add_connection(pml_mesh[5], pml_mesh[2]);
            ctx->atlas().add_connection(pml_mesh[5], pml_mesh[3]);
            ctx->add_domain_as<pml_type>(pml_mesh[5])->setup_center_domain(center_mesh->box()).deploy();

        }
    }
}

}//namespace simpla