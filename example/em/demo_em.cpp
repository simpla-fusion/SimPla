//
// Created by salmon on 16-6-29.
//


#include "../../src/simulation/Context.h"
#include "../../scenario/problem_domain/EMFluid.h"
#include "../../scenario/problem_domain/PML.h"
#include "../../src/manifold/pre_define/PreDefine.h"


namespace simpla
{

void create_scenario(simulation::Context *ctx, ConfigParser const &options)
{
    typedef simpla::manifold::CartesianManifold mesh_type;

    auto center_mesh = ctx->add_mesh<mesh_type>();
    center_mesh->name("Center");
    center_mesh->dimensions(options["Mesh"]["Dimensions"].template as<index_tuple>(index_tuple{20, 20, 1}));
    center_mesh->box(options["Mesh"]["Box"].template as<box_type>(box_type{{0, 0, 0},
                                                                           {1, 1, 1}}));
    center_mesh->ghost_width(index_tuple{2, 2, 2});
    center_mesh->deploy();

    auto center_domain = ctx->add_domain_to<EMFluid<mesh_type >>(center_mesh->id());

    for (auto const &item:options["Particles"])
    {
        auto sp = center_domain->add_particle(std::get<0>(item).template as<std::string>(),
                                              std::get<1>(item)["Charge"].template as<Real>(1.0),
                                              std::get<1>(item)["Mass"].template as<Real>(1.0));
        sp->rho.clear();
        sp->J.clear();
        if (std::get<1>(item)["Density"])
        {
            auto r = center_mesh->range(std::get<1>(item)["Box"].as<box_type>(), VERTEX);
            std::function<Real(point_type const &)> g_obj;
            std::get<1>(item)["Shape"].as(&g_obj);

            std::function<Real(point_type const &)> density;
            std::get<1>(item)["Density"].as(&density);
            r.foreach([&](MeshEntityId const &s)
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
            center_mesh->range(FACE).foreach(
                [&](mesh::MeshEntityId const &s)
                {
                    center_domain->B0[s] = center_mesh->
                        template sample<FACE>(s, fun(center_mesh->point(s)));
                });
        }

        if (options["InitValue"]["B1"])
        {
            std::function<vector_type(point_type const &)> fun;
            options["InitValue"]["B1"]["Value"].as(&fun);
            center_mesh->range(FACE).foreach(
                [&](mesh::MeshEntityId const &s)
                {
                    center_domain->B[s] = center_mesh->template sample<FACE>(s, fun(center_mesh->point(s)));
                });
        }

        if (options["InitValue"]["E1"])
        {
            std::function<vector_type(point_type const &)> fun_fun;
            options["InitValue"]["E1"]["Value"].as(&fun_fun);
            center_mesh->range(EDGE).foreach(
                [&](mesh::MeshEntityId const &s)
                {
                    center_domain->E[s] = center_mesh->template sample<EDGE>(s, fun_fun(center_mesh->point(s)));
                });
        }
    }


    if (options["Constraints"]["J"])
    {

        center_domain->J_src_range = center_mesh->range(
            options["Constraints"]["J"]["Box"].as<box_type>(), mesh::EDGE);

        options["Constraints"]["J"]["Value"].as(&center_domain->J_src_fun);
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

    if (options["PML"])
    {
        typedef PML<mesh_type> pml_type;
        std::shared_ptr<mesh_type> pml_mesh[6];
        index_type w = options["PML"]["Width"].as<index_type>(5);
        index_tuple dims = center_mesh->dimensions();
        index_tuple gw = center_mesh->ghost_width();

        pml_mesh[0] = center_mesh->clone_as<mesh_type>("PML_0");
        pml_mesh[0]->shift(index_tuple{-w, -w, -w});
        pml_mesh[0]->stretch(index_tuple{w, dims[1] + 2 * w, dims[2] + 2 * w});
        pml_mesh[0]->deploy();
        ctx->atlas().add_block(pml_mesh[0]);
        ctx->atlas().add_adjacency2(center_mesh.get(), pml_mesh[0].get(), SP_MB_SYNC);
        ctx->add_domain_as<pml_type>(pml_mesh[0].get())->setup_center_domain(center_mesh->box()).deploy();

        pml_mesh[1] = center_mesh->clone_as<mesh_type>("PML_1");
        pml_mesh[1]->shift(index_tuple{dims[0], -w, -w});
        pml_mesh[1]->stretch(index_tuple{w, dims[1] + 2 * w, dims[2] + 2 * w});
        pml_mesh[1]->deploy();
        ctx->atlas().add_block(pml_mesh[1]);
        ctx->atlas().add_adjacency2(center_mesh.get(), pml_mesh[1].get(), SP_MB_SYNC);
        ctx->add_domain_as<pml_type>(pml_mesh[1].get())->setup_center_domain(center_mesh->box()).deploy();

        if (dims[1] > 1 && gw[1] > 0)
        {
            pml_mesh[2] = center_mesh->clone_as<mesh_type>("PML_2");
            pml_mesh[2]->shift(index_tuple{0, -w, -w});
            pml_mesh[2]->stretch(index_tuple{dims[0], w, dims[2] + 2 * w});
            pml_mesh[2]->deploy();
            ctx->atlas().add_block(pml_mesh[2]);
            ctx->atlas().add_adjacency2(pml_mesh[2].get(), center_mesh.get(), SP_MB_SYNC);
            ctx->atlas().add_adjacency2(pml_mesh[2].get(), pml_mesh[0].get(), SP_MB_SYNC);
            ctx->atlas().add_adjacency2(pml_mesh[2].get(), pml_mesh[1].get(), SP_MB_SYNC);
            ctx->add_domain_as<pml_type>(pml_mesh[2].get())->setup_center_domain(center_mesh->box()).deploy();


            pml_mesh[3] = center_mesh->clone_as<mesh_type>("PML_3");
            pml_mesh[3]->shift(index_tuple{0, dims[1], -w});
            pml_mesh[3]->stretch(index_tuple{dims[0], w, dims[2] + 2 * w});
            pml_mesh[3]->deploy();
            ctx->atlas().add_block(pml_mesh[3]);
            ctx->atlas().add_adjacency2(pml_mesh[3].get(), center_mesh.get(), SP_MB_SYNC);
            ctx->atlas().add_adjacency2(pml_mesh[3].get(), pml_mesh[0].get(), SP_MB_SYNC);
            ctx->atlas().add_adjacency2(pml_mesh[3].get(), pml_mesh[1].get(), SP_MB_SYNC);
            ctx->add_domain_as<pml_type>(pml_mesh[3].get())->setup_center_domain(center_mesh->box()).deploy();

        }
        if (dims[2] > 1 && gw[1] > 0)
        {
            pml_mesh[4] = center_mesh->clone_as<mesh_type>("PML_4");
            pml_mesh[4]->shift(index_tuple{0, 0, -w});
            pml_mesh[4]->stretch(index_tuple{dims[0], dims[1], w});
            pml_mesh[4]->deploy();
            ctx->atlas().add_block(pml_mesh[4]);
            ctx->atlas().add_adjacency2(pml_mesh[4].get(), center_mesh.get(), SP_MB_SYNC);
            ctx->atlas().add_adjacency2(pml_mesh[4].get(), pml_mesh[0].get(), SP_MB_SYNC);
            ctx->atlas().add_adjacency2(pml_mesh[4].get(), pml_mesh[1].get(), SP_MB_SYNC);
            ctx->atlas().add_adjacency2(pml_mesh[4].get(), pml_mesh[2].get(), SP_MB_SYNC);
            ctx->atlas().add_adjacency2(pml_mesh[4].get(), pml_mesh[3].get(), SP_MB_SYNC);
            ctx->add_domain_as<pml_type>(pml_mesh[4].get())->setup_center_domain(center_mesh->box()).deploy();

            pml_mesh[5] = center_mesh->clone_as<mesh_type>("PML_5");
            pml_mesh[5]->shift(index_tuple{0, 0, dims[2]});
            pml_mesh[5]->stretch(index_tuple{dims[0], dims[1], w});
            pml_mesh[5]->deploy();
            ctx->atlas().add_block(pml_mesh[5]);
            ctx->atlas().add_adjacency2(pml_mesh[5].get(), center_mesh.get(), SP_MB_SYNC);
            ctx->atlas().add_adjacency2(pml_mesh[5].get(), pml_mesh[0].get(), SP_MB_SYNC);
            ctx->atlas().add_adjacency2(pml_mesh[5].get(), pml_mesh[1].get(), SP_MB_SYNC);
            ctx->atlas().add_adjacency2(pml_mesh[5].get(), pml_mesh[2].get(), SP_MB_SYNC);
            ctx->atlas().add_adjacency2(pml_mesh[5].get(), pml_mesh[3].get(), SP_MB_SYNC);
            ctx->add_domain_as<pml_type>(pml_mesh[5].get())->setup_center_domain(center_mesh->box()).deploy();

        }
    }
}

}//namespace simpla