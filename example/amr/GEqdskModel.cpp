//
// Created by salmon on 16-11-28.
//


#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/nTupleExt.h>

#include <simpla/model/Model.h>
#include <simpla/manifold/Chart.h>

#include <meshkit/MKCore.hpp>
#include <meshkit/MeshOp.hpp>
#include <meshkit/EBMesher.hpp>
#include <meshkit/SCDMesh.hpp>
#include <meshkit/ModelEnt.hpp>
#include <simpla/model/GEqdsk.h>

namespace simpla { namespace model
{


class GEqdskModel : public model::Model
{
    typedef GEqdskModel this_type;
    typedef model::Model base_type;

public:

    GEqdskModel();

    virtual ~GEqdskModel();

    virtual void update();

    virtual void initialize(Real data_time);

    virtual void load(std::string const &);

    virtual void save(std::string const &);

    virtual void connect(Chart *);

private:
    GEqdsk geqdsk;

    Bundle<Real, VERTEX, 9> m_volume_frac_{"volume", "INPUT"};
    Bundle<Real, VERTEX, 9> m_dual_volume_frac_{"dual_volume", "INPUT"};

};

GEqdskModel::GEqdskModel() : base_type() {}

GEqdskModel::~GEqdskModel() {}


void GEqdskModel::load(std::string const &input_filename)
{
    VERBOSE << " Load " << input_filename << std::endl;

    geqdsk.load(input_filename);
}

void GEqdskModel::connect(Chart *c)
{
    m_volume_frac_.connect(c);
    m_dual_volume_frac_.connect(c);
}


void GEqdskModel::save(std::string const &output_filename)
{
    UNIMPLEMENTED;
}

void GEqdskModel::update() { base_type::update(); }


void
GEqdskModel::initialize(Real data_time)
{
    auto m_start_ = static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 3> *>(m_volume_frac_.data_block())->start();
    auto m_count_ = static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 3> *>(m_volume_frac_.data_block())->count();

    index_type ib = m_start_[0];
    index_type ie = m_start_[0] + m_count_[0];
    index_type jb = m_start_[1];
    index_type je = m_start_[1] + m_count_[1];
    index_type kb = m_start_[2];
    index_type ke = m_start_[2] + m_count_[2];


    for (index_type i = ib; i < ie; ++i)
        for (index_type j = jb; j < je; ++j)
            for (index_type k = kb; k < ke; ++k)
            {
                auto x = get_mesh()->point(i, j, k);
            }

}


}}//namespace simpla //namespace model

namespace simpla
{
std::shared_ptr<model::Model> create_model() { return std::make_shared<model::GEqdskModel>(); }

}