//
// Created by salmon on 16-11-27.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/model/Model.h>
#include <simpla/manifold/Chart.h>

#include <meshkit/MKCore.hpp>
#include <meshkit/MeshOp.hpp>
#include <meshkit/EBMesher.hpp>
#include <meshkit/SCDMesh.hpp>
#include <meshkit/ModelEnt.hpp>

namespace simpla
{
namespace model
{
using namespace MeshKit;

class MeshKitModel;


class MeshKitModel : public model::Model
{
    typedef MeshKitModel this_type;
    typedef model::Model base_type;

public:

    MeshKitModel(ChartBase *c);

    virtual ~MeshKitModel();

    virtual void deploy();

    virtual void initialize(Real data_time = 0);

    virtual void load(std::string const &);

    virtual void save(std::string const &);

private:
    MKCore mk;
    std::string m_input_file_name_;
};

MeshKitModel::MeshKitModel(ChartBase *c) : base_type(c)
{
}

MeshKitModel::~MeshKitModel()
{
    mk.clear_graph();
}


void MeshKitModel::load(std::string const &input_filename)
{
    m_input_file_name_ = input_filename;
}

void MeshKitModel::save(std::string const &output_filename)
{
    mk.save_mesh(output_filename.c_str());
}

void MeshKitModel::deploy() { base_type::deploy(); }

void MeshKitModel::initialize(Real data_time)
{
    base_type::initialize(data_time);

    mk.load_mesh(m_input_file_name_.c_str(), NULL, 0, 0, 0, true);
    // start up MK and load the geometry

    // get the volumes
    MEntVector vols;
    mk.get_entities_by_dimension(3, vols);

    // make EBMesher
    EBMesher *ebm = (EBMesher *) mk.construct_meshop("EBMesher", vols);
    ebm->use_whole_geom(true);
    ebm->use_mesh_geometry(false);

    index_tuple lower, upper;
    std::tie(lower, upper) = chart->coordinate_frame()->mesh_block()->global_index_box();

    int n_interval[3] = {static_cast<int>(upper[0] - lower[0]),
                         static_cast<int>(upper[1] - lower[1]),
                         static_cast<int>(upper[2] - lower[2])};

    ebm->set_num_interval(n_interval);
    ebm->increase_box(0.03); //optional argument. Cartesian mesh box increase form geometry. default 0.03"

    // mesh embedded boundary mesh, by calling execute
    mk.setup_and_execute();

    ebm->export_mesh("output.vtk");

//    if (whole_geom && debug_EBMesher)
    {
        // techX query function test
        double boxMin[3], boxMax[3];
        int nDiv[3];
        std::map<CutCellSurfEdgeKey, std::vector<double>, LessThan> mdCutCellSurfEdge;
        std::vector<int> vnInsideCellTechX;

        ebm->get_grid_and_edges_techX(boxMin, boxMax, nDiv, mdCutCellSurfEdge, vnInsideCellTechX);

        // multiple intersection EBMesh_cpp_fraction query test
        std::map<CutCellSurfEdgeKey, std::vector<double>, LessThan> mdCutCellEdge;
        std::vector<int> vnInsideCell;
        ebm->get_grid_and_edges(boxMin, boxMax, nDiv, mdCutCellEdge, vnInsideCell);

        VERBOSE << SHORT_FILE_LINE_STAMP << "# of TechX cut-cell surfaces: " << mdCutCellSurfEdge.size()
                << ", # of nInsideCell: " << vnInsideCell.size() / 3 << std::endl;
    }


}

}  //namespace model

std::shared_ptr<model::Model>
create_modeler(mesh::ChartBase *chart, std::string const &input_file_name = "")
{
    auto res = std::make_shared<model::MeshKitModel>(chart);
    res->load(input_file_name);
    return std::dynamic_pointer_cast<model::Model>(res);
}
}//namespace simpla