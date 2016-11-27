//
// Created by salmon on 16-11-27.
//
#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/nTupleExt.h>

#include "meshkit/MKCore.hpp"
#include "meshkit/MeshOp.hpp"
#include "meshkit/EBMesher.hpp"
#include "meshkit/SCDMesh.hpp"
#include "meshkit/ModelEnt.hpp"

using namespace MeshKit;
namespace simpla
{
void step2vtk(std::string const &input_filename, std::string const &output_filename = "")
{
    bool result;
    time_t start_time, load_time, mesh_time, vol_frac_time,
            export_time, query_time_techX, query_time;

    // start up MK and load the geometry
    MKCore mk;

    mk.load_mesh(input_filename.c_str(), NULL, 0, 0, 0, true);

    mk.save_mesh((input_filename + ".vtk").c_str());

    // get the volumes
    MEntVector vols;
    mk.get_entities_by_dimension(3, vols);

    // make EBMesher
    EBMesher *ebm = (EBMesher *) mk.construct_meshop("EBMesher", vols);
    ebm->use_whole_geom(whole_geom);
    ebm->use_mesh_geometry(mesh_based_geom);
    ebm->set_num_interval(n_interval);
    ebm->increase_box(box_increase);
    if (mesh_based_geom) ebm->set_obb_tree_box_dimension();

    // mesh embedded boundary mesh, by calling execute
    mk.setup_and_execute();
    time(&mesh_time);

    // caculate volume fraction, only for geometry input
    if (vol_frac_res > 0)
    {
        result = ebm->get_volume_fraction(vol_frac_res);
        if (!result)
        {
            std::cerr << "Couldn't get volume fraction." << std::endl;
            return 1;
        }
    }
    time(&vol_frac_time);

    // export mesh
    if (output_filename != NULL)
    {
        ebm->export_mesh(output_filename);
    }
    time(&export_time);


    // techX query function test
    double boxMin[3], boxMax[3];
    int nDiv[3];
    std::map<CutCellSurfEdgeKey, std::vector<double>, LessThan> mdCutCellSurfEdge;
    std::vector<int> vnInsideCellTechX;

    ebm->get_grid_and_edges_techX(boxMin, boxMax, nDiv,
                                  mdCutCellSurfEdge, vnInsideCellTechX);
    time(&query_time_techX);

    // multiple intersection EBMesh_cpp_fraction query test
    std::map<CutCellSurfEdgeKey, std::vector<double>, LessThan> mdCutCellEdge;
    std::vector<int> vnInsideCell;
    result = ebm->get_grid_and_edges(boxMin, boxMax, nDiv,
                                     mdCutCellEdge, vnInsideCell);
    if (!result)
    {
        std::cerr << "Couldn't get mesh information." << std::endl;
        return 1;
    }
    time(&query_time);
    std::cout << "# of TechX cut-cell surfaces: " << mdCutCellSurfEdge.size()
              << ", # of nInsideCell: " << vnInsideCell.size() / 3 << std::endl;


    std::cout << "EBMesh is succesfully finished." << std::endl;
    std::cout << "Time including loading: "
              << difftime(mesh_time, start_time)
              << " secs, Time excluding loading: "
              << difftime(mesh_time, load_time)
              << " secs, Time volume fraction: "
              << difftime(vol_frac_time, mesh_time) << " secs";

    if (output_filename != NULL)
    {
        std::cout << ", Time export mesh: " << difftime(export_time, vol_frac_time) << " secs";
    }


    std::cout << ", TechX query time: "
              << difftime(query_time_techX, export_time)
              << " secs, multiple intersection EBMesh_cpp_fraction query (elems, edge-cut fractions): "
              << difftime(query_time, query_time_techX) << " secs.";


    std::cout << std::endl;
    mk.clear_graph();

    return 0;
}
}