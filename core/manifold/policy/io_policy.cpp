/**
 * @file io_policy.cpp
 * @author salmon
 * @date 2015-12-02.
 */

#include <string>
#include "io_policy.h"

#include "../../dataset/dataset.h"
#include "../../gtl/utilities/log.h"
#include "../../gtl/type_cast.h"

#ifndef NO_XDMF

#include "../../io/xdmf_io.h"
#include "../../io/io.h"

#endif

namespace simpla { namespace manifold { namespace policy
{
struct MeshIOBase::pimpl_s
{
    pimpl_s();

    ~pimpl_s();


    virtual bool read();

    virtual void write();

    virtual void set_time(Real t);

    virtual void register_dataset(std::string const &url, DataSet const &ds, int IFORM = 0);


    XdmfDOM m_dom_;

    XdmfRoot m_root_;

    XdmfDomain m_domain_;

    XdmfGrid m_grid_;

    Real m_time_;

    std::string m_grid_name_;


    std::string m_prefix_;


    int m_grid_type_id_;

    std::string m_topology_type_str_;

    std::map<std::string, std::tuple<int, DataSet>> m_datasets_;


    std::string m_xdmf_txt_;
};

MeshIOBase::pimpl_s::pimpl_s()
{
    m_root_.SetDOM(&m_dom_);
    m_root_.SetVersion(2.0);
    m_root_.Build();
    m_root_.Insert(&m_domain_);
    m_domain_.Insert(&m_grid_);
}

MeshIOBase::pimpl_s::~pimpl_s()
{

}

void MeshIOBase::pimpl_s::set_time(Real t)
{
    m_time_ = t;
//    m_grid_.SetTime(t);
}

/**
 *
 *
 *    XML Attribute : TopologyType = Polyvertex | Polyline | Polygon |
                                  Triangle | Quadrilateral | Tetrahedron | Pyramid| Wedge | Hexahedron |
                                  Edge_3 | Triangle_6 | Quadrilateral_8 | Tetrahedron_10 | Pyramid_13 |
                                  Wedge_15 | Wedge_18 | Hexahedron_20 | Hexahedron_24 | Hexahedron_27 |
                                  Mixed |
                                  2DSMesh | 2DRectMesh | 2DCoRectMesh |
                                  3DSMesh | 3DRectMesh | 3DCoRectMesh
 */

void  MeshIOBase::deploy(int ndims, size_t const *pdims, Real const *xmin, Real const *dx)
{
    m_pimpl_->m_grid_.SetGridType(XDMF_GRID_UNIFORM);

    std::string topology_type_str;

    nTuple<XdmfInt64, 3> dims;

    dims = pdims;

    if (ndims == 2)
    {

        m_pimpl_->m_grid_.GetTopology()->SetTopologyTypeFromString("2DCoRectMesh");
        m_pimpl_->m_grid_.GetTopology()->GetShapeDesc()->SetShape(ndims, &dims[0]);
        m_pimpl_->m_grid_.GetGeometry()->SetGeometryTypeFromString("Origin_DxDy");
        m_pimpl_->m_grid_.GetGeometry()->SetOrigin(xmin[0], xmin[1], 0);
        m_pimpl_->m_grid_.GetGeometry()->SetDxDyDz(dx[0], dx[1], 0);
    }
    else if (ndims == 3)
    {
        m_pimpl_->m_grid_.GetTopology()->SetTopologyTypeFromString("3DCoRectMesh");
        m_pimpl_->m_grid_.GetTopology()->GetShapeDesc()->SetShape(ndims, &dims[0]);
        m_pimpl_->m_grid_.GetGeometry()->SetGeometryTypeFromString("Origin_DxDyDz");
        m_pimpl_->m_grid_.GetGeometry()->SetOrigin(xmin[0], xmin[1], xmin[2]);
        m_pimpl_->m_grid_.GetGeometry()->SetDxDyDz(dx[0], dx[1], dx[2]);

    }
    else
    {
        THROW_EXCEPTION_RUNTIME_ERROR(" number of dimension is not 2 or 3");
    }

}


void  MeshIOBase::deploy(int ndims, size_t const *pdims, point_type const *points)
{

    m_pimpl_->m_grid_.SetGridType(XDMF_GRID_UNIFORM);

    std::string topology_type_str;

    nTuple<XdmfInt64, 3> dims;

    dims = pdims;

    if (ndims == 3)
    {

        m_pimpl_->m_grid_.GetTopology()->SetTopologyTypeFromString("3DSMesh");
        m_pimpl_->m_grid_.GetTopology()->GetShapeDesc()->SetShape(ndims, &dims[0]);

        m_pimpl_->m_grid_.GetGeometry()->SetGeometryTypeFromString("XYZ");

    }
    else
    {
        THROW_EXCEPTION_RUNTIME_ERROR("unsportted grid type");
    }

}


void MeshIOBase::pimpl_s::register_dataset(std::string const &name, DataSet const &ds, int IFORM)
{
    std::get<0>(m_datasets_[name]) = IFORM;

    DataSet(ds).swap(std::get<1>(m_datasets_[name]));

}

bool MeshIOBase::pimpl_s::read()
{
    UNIMPLEMENTED;
    return false;
}

void  MeshIOBase::pimpl_s::write()
{


    VERBOSE << "write XDMF [" << m_prefix_ << ".xdmf/" << m_grid_name_ << "/" << "]" << std::endl;

    XdmfInformation info;
    info.SetName("Time");
    info.SetValue(type_cast<std::string>(m_time_).c_str());
    m_grid_.Insert(&info);


    XdmfAttribute myAttribute;
    m_grid_.Insert(&myAttribute);


    std::vector<XdmfDataItem> data(m_datasets_.size());
    int count = 0;
    for (auto const &item:m_datasets_)
    {


        std::string name = item.first;
        int IFORM = std::get<0>(item.second);
        DataSet const &ds = std::get<1>(item.second);

        VERBOSE << "write XDMF [" << m_prefix_ << ".xdmf/" << m_grid_name_ << "/" << name << "]" << std::endl;


        myAttribute.SetName(item.first.c_str());

        if (ds.datatype.is_array())
        {
            myAttribute.SetAttributeTypeFromString("Vector");
        }
        else
        {

            myAttribute.SetAttributeTypeFromString("Scalar");
        }

        switch (IFORM)
        {
            case 0:
                myAttribute.SetAttributeCenterFromString("Node");
                break;

            case 1:
                myAttribute.SetAttributeCenterFromString("Edge");
                break;
            case 2:
                myAttribute.SetAttributeCenterFromString("Face");
                break;
            case 3:
                myAttribute.SetAttributeCenterFromString("Cell");
                break;

            default:
                THROW_EXCEPTION_RUNTIME_ERROR("IFORM >3")
        }


        myAttribute.Insert(&data[count]);


        int ndims = 3;

        nTuple<XdmfInt64, 4> dims;

        std::tie(ndims, dims, std::ignore, std::ignore, std::ignore, std::ignore) = ds.dataspace.shape();

        io::cd(m_prefix_ + ".h5:/" + m_grid_name_ + "/");

        auto url = io::save(name, ds);

        data[count].SetHeavyDataSetName(url.c_str());
        data[count].SetShape(ndims, (&dims[0]));
        data[count].SetFormat(XDMF_FORMAT_HDF);
        data[count].SetArrayIsMine(false);

        ++count;
    }

    m_grid_.Build();

    std::ofstream ss(m_prefix_ + ".xdmf");

    ss << m_dom_.Serialize() << std::endl;
}

MeshIOBase::MeshIOBase() : m_pimpl_(new pimpl_s)
{
    set_io_prefix();
}

MeshIOBase::~MeshIOBase()
{

}

void  MeshIOBase::write() const
{
    m_pimpl_->write();
}


bool  MeshIOBase::read()
{
    return m_pimpl_->read();
}


void MeshIOBase::register_dataset(std::string const &name, DataSet const &ds, int IFORM)
{
    m_pimpl_->register_dataset(name, ds, IFORM);

}

void MeshIOBase::set_io_prefix(std::string const &prefix, const std::string &name)
{
    m_pimpl_->m_prefix_ = prefix;
    m_pimpl_->m_grid_name_ = name;

}


void MeshIOBase::set_io_time(Real t)
{
    m_pimpl_->set_time(t);
}
}}}//namespace simpla { namespace manifold { namespace policy
