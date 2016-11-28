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

    MeshKitModel();

    virtual ~MeshKitModel();

    virtual void deploy();

    virtual void initialize(Real data_time = 0);

    virtual void load(std::string const &);

    virtual void save(std::string const &);

private:
    std::string m_input_file_name_;

    MKCore mk;
    std::shared_ptr<EBMesher> ebm;
    MEntVector vols;


};

MeshKitModel::MeshKitModel() : base_type()
{
}

MeshKitModel::~MeshKitModel()
{
    ebm.reset();
    mk.clear_graph();

}


void MeshKitModel::load(std::string const &input_filename)
{
    m_input_file_name_ = input_filename;

    VERBOSE << " Load " << m_input_file_name_ << std::endl;
    mk.load_mesh(m_input_file_name_.c_str(), NULL, 0, 0, 0, true);
    // start up MK and load the geometry

    // get the volumes
    mk.get_entities_by_dimension(3, vols);

    // make EBMesher
    ebm = std::shared_ptr<EBMesher>((EBMesher *) mk.construct_meshop("EBMesher", vols));
    ebm->use_whole_geom(true);
    ebm->use_mesh_geometry(false);

//    index_tuple lower, upper;
//    std::tie(lower, upper) = chart->coordinate_frame()->mesh_block()->global_index_box();
//
//    int n_interval[3] = {static_cast<int>(upper[0] - lower[0]),
//                         static_cast<int>(upper[1] - lower[1]),
//                         static_cast<int>(upper[2] - lower[2])};
    int n_interval[3] = {64, 64, 64};
    ebm->set_num_interval(n_interval);
    ebm->increase_box(0.03); //optional argument. Cartesian mesh box increase form geometry. default 0.03"

    // mesh embedded boundary mesh, by calling execute
    mk.setup_and_execute();

}

void MeshKitModel::save(std::string const &output_filename)
{
    ASSERT(output_filename != "");
    mk.save_mesh(output_filename.c_str());
    ebm->export_mesh(output_filename.c_str(), true);

}

void MeshKitModel::deploy() { base_type::deploy(); }

void MeshKitModel::initialize(Real data_time)
{
    base_type::initialize(data_time);

//    point_type lower, upper;
//    std::tie(lower, upper) = chart->coordinate_frame()->box();
//
//    index_tuple i_lower, i_upper;
//    std::tie(i_lower, i_upper) = chart->coordinate_frame()->mesh_block()->inner_index_box();
//
//    int nDiv[3] = {i_upper[0] - i_lower[0], i_upper[1] - i_lower[1], i_upper[2] - i_lower[2]};
//
//    std::map<CutCellSurfEdgeKey, std::vector<double>, LessThan> mdCutCellSurfEdge;
//
//    std::vector<int> vnInsideCellTechX;
//
//    ebm->get_grid_and_edges_techX(&(lower[0]), &(upper[0]), nDiv, mdCutCellSurfEdge, vnInsideCellTechX);

//        // multiple intersection EBMesh_cpp_fraction query test
//        std::map<CutCellSurfEdgeKey, std::vector<double>, LessThan> mdCutCellEdge;
//        std::vector<int> vnInsideCell;
//        ebm->get_grid_and_edges(boxMin, boxMax, nDiv, mdCutCellEdge, vnInsideCell);
//
//    VERBOSE << SHORT_FILE_LINE_STAMP << "# of TechX cut-cell surfaces: " << mdCutCellSurfEdge.size()
//            << ", # of nInsideCell: " << vnInsideCellTechX.size() / 3 << std::endl;


}

}  //namespace model

std::shared_ptr<model::Model>
create_model(const std::string &input_file_name)
{
    auto res = std::make_shared<model::MeshKitModel>();
    res->load(input_file_name);
    return std::dynamic_pointer_cast<model::Model>(res);
}
}//namespace simpla