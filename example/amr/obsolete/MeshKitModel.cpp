//
// Created by salmon on 16-11-27.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/nTupleExt.h>

#include <simpla/model/Model.h>
#include <simpla/mesh/Mesh.h>

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
    std::map<CutCellSurfEdgeKey, std::vector<double>, LessThan> mdCutCellSurfEdge;
    std::vector<int> vnInsideCellTechX;
    point_type x_min, x_max;
    index_tuple m_count_;
};

MeshKitModel::MeshKitModel() : base_type() {}

MeshKitModel::~MeshKitModel() { mk.clear_graph(); }


void MeshKitModel::load(std::string const &input_filename)
{
    VERBOSE << " Load " << input_filename << std::endl;
    index_tuple i_lower, i_upper;

    std::tie(i_lower, i_upper) = db["global index box"].as<index_box_type>();

    int n_interval[3] = {static_cast<int>(i_upper[0] - i_lower[0]),
                         static_cast<int>(i_upper[1] - i_lower[1]),
                         static_cast<int>(i_upper[2] - i_lower[2])};

    mk.load_mesh(input_filename.c_str(), NULL, 0, 0, 0, true);
    mk.save_mesh((input_filename + ".vtk").c_str());

    mk.get_entities_by_dimension(3, m_vols_);

    //    EBMesher ebm(&mk, m_vols_);
//
//    ebm.set_num_interval(n_interval);
//
//    ebm.set_division(&std::get<0>(bound_box)[0], &std::get<1>(bound_box)[0]);
//
//    ebm.increase_box(0.03);     //optional argument. Cartesian mesh box increase form geometry. default 0.03"
//
//    ebm.setup_this();
//    // mesh embedded boundary mesh, by calling execute
//    ebm.execute_this();


    EBMesher *ebm = (EBMesher *) mk.construct_meshop("EBMesher", m_vols_);
    ebm->use_whole_geom(1);
    ebm->use_mesh_geometry(0);
    ebm->set_num_interval(n_interval);
    ebm->increase_box(0.03);

    // mesh embedded boundary mesh, by calling execute
    mk.setup_and_execute();

    double boxMin[3], boxMax[3];
    int nDiv[3];


    ebm->get_grid_and_edges_techX(boxMin, boxMax, nDiv, mdCutCellSurfEdge, vnInsideCellTechX);
    x_min[0] = boxMin[0];
    x_min[1] = boxMin[1];
    x_min[2] = boxMin[2];
    x_max[0] = boxMax[0];
    x_max[1] = boxMax[1];
    x_max[2] = boxMax[2];
    m_count_[0] = nDiv[0];
    m_count_[1] = nDiv[1];
    m_count_[2] = nDiv[2];

    VERBOSE << "{" << x_min << "," << x_max << "} [" << m_count_ << "]" << std::endl;

    ebm->export_mesh("ebmesh.vtk", true);
    delete ebm;
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
}


}  //namespace model

std::shared_ptr<model::Model> create_model() { return std::make_shared<model::MeshKitModel>(); }
}//namespace simpla