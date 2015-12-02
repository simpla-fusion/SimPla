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

    virtual void write(Real t) const;


    virtual void register_dataset(std::string const &url, DataSet const &ds, int IFORM = 0);

    std::map<std::string, std::tuple<int, DataSet>> m_datasets_;


    std::string m_grid_name_;

    std::string m_prefix_;

    std::string m_place_holder_;

    std::string m_file_contents_;

};

MeshIOBase::pimpl_s::pimpl_s()
{


    m_place_holder_ = "<!-- PLACE HOLDER -->";

    m_file_contents_ = ""
                               "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2\">\n"
                               "  <Domain>\n"
                               "    <Grid GridType=\"Uniform\">\n"
                       + m_place_holder_ + "\n"
                               "    </Grid>\n"
                               "  </Domain>\n"
                               "</Xdmf>";


}

MeshIOBase::pimpl_s::~pimpl_s()
{

}


void _str_replace(std::string *s, std::string const &place_holder, std::string const &txt)
{
    s->replace(s->find(place_holder), place_holder.size(), txt + "\n" + place_holder);
}

std::string save_dataitem(std::string const &prefix, std::string const ds_name, DataSet const &ds)
{
    io::cd(prefix);

    std::string url = io::save(ds_name, ds);

    VERBOSE << "write data item [" << url << "/" << "]" << std::endl;

    std::ostringstream buffer;

    int ndims;

    nTuple<size_t, MAX_NDIMS_OF_ARRAY> dims;

    std::tie(ndims, dims, std::ignore, std::ignore, std::ignore, std::ignore) = ds.dataspace.shape();

    buffer
    << "\t\t  <DataItem Dimensions=\"";
    for (int i = 0; i < ndims; ++i)
    {
        buffer << dims[i] << " ";
    }

    buffer << "\" "
    << "NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n"
    << "\t\t   " << url << std::endl
    << "\t\t  </DataItem>\n";
    return buffer.str();
}

template<typename T>
std::string save_dataitem(std::string const &prefix, std::string const ds_name, size_t num, T const *p)
{

    io::cd(prefix);

    std::string url = io::save(ds_name, num, p);

    VERBOSE << "write data item [" << url << "/" << "]" << std::endl;

    io::cd(prefix);

    std::ostringstream buffer;


    buffer
    << "\t <DataItem Dimensions=\"" << num << "\" " << "NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n"
    << url << std::endl
    << "\t</DataItem>";

    return buffer.str();
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

void  MeshIOBase::dump_grid(int ndims, size_t const *dims, Real const *xmin, Real const *dx)
{


    std::ostringstream buffer;


    if (ndims == 3)
    {
        buffer << ""
                "\t <Topology TopologyType=\"3DCoRectMesh\""
                "\t      Dimensions=\"" << dims[0] << " " << dims[1] << " " << dims[2] << "\"/>\n"
                "\t <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
                "\t   <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" "
                " Precision=\"4\" Format=\"XML\">" << xmin[0] << " " << xmin[1] << " " << xmin[2] << "</DataItem>\n"
                "\t   <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\""
                " Precision=\"4\" Format=\"XML\">" << dx[0] << " " << dx[1] << " " << dx[2] << "  </DataItem >\n"
                "\t </Geometry>";


    }
    else if (ndims == 2)
    {
        buffer << ""
                "\t <Topology TopologyType=\"2DCoRectMesh\""
                "\t      Dimensions=\"" << dims[0] << " " << dims[1] << "\"/>\n"
                "\t <Geometry GeometryType=\"ORIGIN_DXDY\">\n"
                "\t   <DataItem Name=\"Origin\" Dimensions=\"2\" NumberType=\"Float\" "
                " Precision=\"4\" Format=\"XML\">" << xmin[0] << " " << xmin[1] << "</DataItem>\n"
                "\t   <DataItem Name=\"Spacing\" Dimensions=\"2\" NumberType=\"Float\""
                " Precision=\"4\" Format=\"XML\">" << dx[0] << " " << dx[1] << "  </DataItem >\n"
                "\t </Geometry>";


    }
    else
    {
        THROW_EXCEPTION_RUNTIME_ERROR(" number of dimension is not 2 or 3");
    }


    _str_replace(&(m_pimpl_->m_file_contents_), m_pimpl_->m_place_holder_, buffer.str());

}


void  MeshIOBase::dump_grid(DataSet const &ds)
{

    io::cd(m_pimpl_->m_prefix_ + ".h5:/");

    std::ostringstream buffer;

    int ndims;

    nTuple<size_t, MAX_NDIMS_OF_ARRAY> dims;

    std::tie(ndims, dims, std::ignore, std::ignore, std::ignore, std::ignore) = ds.dataspace.shape();

    --ndims;

    if (ndims == 2)
    {
        buffer << ""
        << "\t <Topology TopologyType=\"2DSMesh\""
        << "\t      NumberOfElements=\"" << dims[0] << " " << dims[1] << " " << dims[2] << "\"/>\n"
        << "\t <Geometry GeometryType=\"XY\">\n"
        << save_dataitem("/" + m_pimpl_->m_grid_name_ + "/", "points", ds)
        << "\t </Geometry>\n";
    }

    else if (ndims == 3)
    {
        buffer << ""
        << "\t <Topology TopologyType=\"3DSMesh\""
        << "\t      NumberOfElements=\"" << dims[0] << " " << dims[1] << " " << dims[2] << "\"/>\n"
        << "\t <Geometry GeometryType=\"XYZ\">\n"
        << save_dataitem("/" + m_pimpl_->m_grid_name_ + "/", "points", ds)
        << "\t </Geometry>\n";
    }
    else
    {
        THROW_EXCEPTION_RUNTIME_ERROR("unsportted grid type");
    }
    _str_replace(&(m_pimpl_->m_file_contents_), m_pimpl_->m_place_holder_, buffer.str());
}


void MeshIOBase::pimpl_s::register_dataset(std::string const &name, DataSet const &ds, int IFORM)
{
    std::get<0>(m_datasets_[name]) = IFORM;
    std::get<1>(m_datasets_[name]) = ds;
//
//    ds_.dataspace = ds.dataspace;
//    ds_.memory_space = ds.memory_space;
//    ds_.datatype = ds.datatype;
//    ds_.data = ds.data;

}

bool MeshIOBase::pimpl_s::read()
{
    UNIMPLEMENTED;
    return false;
}


void  MeshIOBase::pimpl_s::write(Real t) const
{


    VERBOSE << "write XDMF [" << m_prefix_ << ".xdmf/" << m_grid_name_ << "/" << "]" << std::endl;


    std::ostringstream buffer;

    buffer << "\t <Information Name=\"Time\" Value=\"" << t << "\"/>" << std::endl;


    int count = 0;
    for (auto const &item:m_datasets_)
    {


        std::string ds_name = item.first;
        int IFORM = std::get<0>(item.second);
        DataSet const &ds = std::get<1>(item.second);

        std::string a_type = (ds.datatype.is_array() || IFORM == 1 || IFORM == 2) ? "Vector" : "Scalar";

//        static const char a_center[][10] = {
//                "Node",
//                "Edge",
//                "Face",
//                "Cell"
//        };

        std::string a_center = "Node";

        buffer << ""
        << "\t <Attribute Name=\"" << ds_name << "\"  AttributeType=\"" << a_type
        << "\" Center=\"" << a_center << "\">\n"
        << save_dataitem("/" + m_grid_name_ + "/", ds_name, std::get<1>(item.second))
        << "\t </Attribute>"
        << std::endl;

    }

    std::string file_content = m_file_contents_;

    _str_replace(&(file_content), m_place_holder_, buffer.str());

    std::ofstream ss(m_prefix_ + ".xdmf");

    ss << file_content << std::endl;
}

MeshIOBase::MeshIOBase() : m_pimpl_(new pimpl_s)
{
    set_prefix();
}

MeshIOBase::~MeshIOBase()
{

}

void  MeshIOBase::write() const
{
    m_pimpl_->write(time());
}


bool  MeshIOBase::read()
{
    return m_pimpl_->read();
}


void MeshIOBase::register_dataset(std::string const &name, DataSet const &ds, int IFORM)
{
    m_pimpl_->register_dataset(name, ds, IFORM);

}

void MeshIOBase::set_prefix(std::string const &prefix, const std::string &name)
{
    m_pimpl_->m_prefix_ = prefix;
    m_pimpl_->m_grid_name_ = name;

}


}}}//namespace simpla { namespace manifold { namespace policy
