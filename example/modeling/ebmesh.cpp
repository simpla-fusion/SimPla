/*!
\example example_ebmesh.cpp
The EBMesh tool can generate Cartesian meshes for solvers that use embedded boundary algorithms. \n
It uses a ray-tracing technique based on hierarchical oriented bounding boxes in MOAB.
\section EBMesh_cpp_title EBMesher Tool With Options

\subsection EBMesh_cpp_in Input
Reads in an input geometry file from data directory (sphere),
\subsection EBMesh_cpp_out Output
Exports a mesh file with CutCell data.

\subsection EBMesh_cpp_src Source Code
*/
#include "meshkit/MKCore.hpp"
#include "meshkit/MeshOp.hpp"
#include "meshkit/EBMesher.hpp"
#include "meshkit/SCDMesh.hpp"
#include "meshkit/ModelEnt.hpp"

using namespace MeshKit;


#define DEFAULT_TEST_FILE "demo.stp"

const bool debug_EBMesher = true;

int load_and_mesh(const char *input_filename,
                  const char *output_filename,
                  int whole_geom, int *n_intervals, int mesh_based_geom,
                  double box_increase, int vol_frac_res);

int main(int argc, char **argv)
{
    // check command line arg
    std::string input_filename;
    const char *output_filename = NULL;
    int whole_geom = 1;
    int n_interval[3] = {100, 100, 20};
    int mesh_based_geom = 0;
    double box_increase = .03;
    int vol_frac_res = 0;

    if (argc > 2 && argc < 11)
    {
        input_filename += argv[1];
        if (argc > 2) whole_geom = atoi(argv[2]);
        if (argc > 3) n_interval[0] = atoi(argv[3]);
        if (argc > 4) n_interval[1] = atoi(argv[4]);
        if (argc > 5) n_interval[2] = atoi(argv[5]);
        if (argc > 6) mesh_based_geom = atoi(argv[6]);
        if (argc > 7) output_filename = argv[7];
        if (argc > 8) box_increase = atof(argv[8]);
        if (argc > 9) vol_frac_res = atoi(argv[9]);
    }
    else
    {
        std::cout << "Usage: " << argv[0] <<
        "<input_geom_filename> <whole_geom> {x: # of intervals} {y: # of intervals} {z: # of intervals} {mesh_based_geom} {output_mesh_filename} {box_size_increase} {vol_frac_res}" <<
        std::endl;
        std::cout << "<input_geom_filename> : input geometry file name" << std::endl;
        std::cout << "<whole_geom> : make mesh for whole geom or individually(1/0), default whole geom(1)" << std::endl;
        std::cout << "{x/y/z: # ofintervals} : optional argument. # of intervals. if it is not set, setValue to 10." <<
        std::endl;
        std::cout << "<mesh_based_geom> : use mesh based geometry(1/0), default not-use(0)" << std::endl;
        std::cout <<
        "{output_mesh_filename} : optional argument. if it is not setValue, dosn't export. can output mesh file (e.g. output.vtk.)" <<
        std::endl;
        std::cout <<
        "{box size increase} : optional argument. Cartesian mesh box increase form geometry. default 0.03" << std::endl;
        std::cout <<
        "{vol_frac_res} : optional argument, volume fraction resolution of boundary cells for each material, you can specify it as # of divisions (e.g. 4)." <<
        std::endl;
        std::cout << std::endl;
        if (argc != 1) return 1;
        std::cout << "No file specified.  Defaulting to: " << DEFAULT_TEST_FILE << std::endl;
        std::string file_name = DEFAULT_TEST_FILE;

        input_filename += DEFAULT_TEST_FILE;
    }

    if (load_and_mesh(input_filename.c_str(), "demo_ebmesh.vtk",
                      whole_geom, n_interval, mesh_based_geom, box_increase, vol_frac_res))
        return 1;

    return 0;
}

int load_and_mesh(const char *input_filename,
                  const char *output_filename,
                  int whole_geom, int *n_interval, int mesh_based_geom,
                  double box_increase, int vol_frac_res)
{
    bool result;
    time_t start_time, load_time, mesh_time, vol_frac_time,
            export_time, query_time_techX, query_time;

    // start up MK and Load the geometry
    MKCore mk;

    mk.load_mesh(input_filename, NULL, 0, 0, 0, true);

    mk.save_mesh("input.vtk");

    // Get the volumes
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
            std::cerr << "Couldn't Get volume fraction." << std::endl;
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
        std::cerr << "Couldn't Get mesh information." << std::endl;
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
