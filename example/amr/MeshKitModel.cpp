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

    virtual void update();

    virtual void initialize(Real data_time);

    virtual void load(std::string const &);

    virtual void save(std::string const &);

private:


    MKCore mk;
    MEntVector m_vols_;
};

MeshKitModel::MeshKitModel() : base_type() {}

MeshKitModel::~MeshKitModel() { mk.clear_graph(); }


void MeshKitModel::load(std::string const &input_filename)
{
    VERBOSE << " Load " << input_filename << std::endl;
    mk.load_mesh(input_filename.c_str(), NULL, 0, 0, 0, true);
    // start up MK and load the geometry

    // get the volumes
    mk.get_entities_by_dimension(3, m_vols_);
}

void MeshKitModel::save(std::string const &output_filename)
{
    ASSERT(output_filename != "");
    mk.save_mesh(output_filename.c_str());

}

void MeshKitModel::update() { base_type::update(); }


void
MeshKitModel::initialize(Real data_time)
{
    // make EBMesher
    EBMesher ebm(&mk, m_vols_);

    auto bound_box = get_chart()->mesh_block()->box();

    index_tuple i_lower, i_upper;

    std::tie(i_lower, i_upper) = get_chart()->mesh_block()->inner_index_box();

    int n_interval[3] = {static_cast<int>(i_upper[0] - i_lower[0]),
                         static_cast<int>(i_upper[1] - i_lower[1]),
                         static_cast<int>(i_upper[2] - i_lower[2])};

    ebm.set_num_interval(n_interval);

    ebm.set_division(&std::get<0>(bound_box)[0], &std::get<0>(bound_box)[1]);

    ebm.increase_box(0.03);     //optional argument. Cartesian mesh box increase form geometry. default 0.03"

    ebm.setup_this();
    // mesh embedded boundary mesh, by calling execute
    ebm.execute_this();


    double boxMin[3], boxMax[3];
    int nDiv[3];
    std::map<CutCellSurfEdgeKey, std::vector<double>, LessThan> mdCutCellSurfEdge;
    std::vector<int> vnInsideCellTechX;

    ebm.get_grid_and_edges_techX(boxMin, boxMax, nDiv, mdCutCellSurfEdge, vnInsideCellTechX);

    ebm.export_mesh("sub_block.vtk", true);

}


}  //namespace model

std::shared_ptr<model::Model> create_model() { return std::make_shared<model::MeshKitModel>(); }
}//namespace simpla