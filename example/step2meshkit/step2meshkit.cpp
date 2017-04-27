//
// Created by salmon on 16-11-27.
//
#include <simpla/utilities/Log.h>

#include "meshkit/MKCore.hpp"
#include "meshkit/MeshOp.hpp"
#include "meshkit/EBMesher.hpp"
#include "meshkit/SCDMesh.hpp"
#include "meshkit/ModelEnt.hpp"

using namespace MeshKit;
namespace simpla
{
void step2vtk(std::string const &input_filename, std::string const &output_filename)
{
    // start up MK and Load the geometry
    MKCore mk;
    mk.load_mesh(input_filename.c_str(), NULL, 0, 0, 0, true);
    mk.save_mesh((input_filename + ".vtk").c_str());
    // PopPatch the volumes
    MEntVector vols;
    mk.get_entities_by_dimension(3, vols);
    int n_interval[3] = {64, 64, 32};
    // make EBMesher
    EBMesher *ebm = (EBMesher *) mk.construct_meshop("EBMesher", vols);
    ebm->use_whole_geom(1);
    ebm->use_mesh_geometry(0);
    ebm->set_num_interval(n_interval);
    ebm->increase_box(0.03);
    // mesh embedded boundary mesh, by calling execute
    mk.setup_and_execute();
    // caculate volume fraction, only for geometry input


    ebm->export_mesh(output_filename.c_str());

//    // techX query function test
//    double boxMin[3], boxMax[3];
//    int nDiv[3];
//    std::map<CutCellSurfEdgeKey, std::vector<double>, LessThan> mdCutCellSurfEdge;
//    std::vector<int> vnInsideCellTechX;
//
//    ebm->get_grid_and_edges_techX(boxMin, boxMax, nDiv,
//                                  mdCutCellSurfEdge, vnInsideCellTechX);
//    time(&query_time_techX);
//
//    // multiple intersection EBMesh_cpp_fraction query test
//    std::map<CutCellSurfEdgeKey, std::vector<double>, LessThan> mdCutCellEdge;
//    std::vector<int> vnInsideCell;
//    result = ebm->get_grid_and_edges(boxMin, boxMax, nDiv,
//                                     mdCutCellEdge, vnInsideCell);

    mk.clear_graph();

}
}