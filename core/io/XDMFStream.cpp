/**
 * @file xdmf_stream.cpp.cpp
 * @author salmon
 * @date 2015-12-02.
 */
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include "XDMFStream.h"

#include "IO.h"
#include "../gtl/utilities/log.h"
#include "../data_model/DataSet.h"
#include "../parallel/Parallel.h"

namespace simpla { namespace io
{


XDMFStream::XDMFStream()
{
    m_prefix_ = "";
}

XDMFStream::~XDMFStream()
{
    close();
}

std::string XDMFStream::path() const
{
    std::ostringstream buffer;
    buffer << "/";
    for (auto const &g_name: m_path_)
    {
        buffer << g_name << "/";
    }
    return buffer.str();
}


void XDMFStream::open(std::string const &prefix, std::string const &grid_name)
{

    close();

    m_prefix_ = prefix;

    m_file_stream_.open(m_prefix_ + ".xdmf");

    m_h5_stream_.open(m_prefix_ + ".h5:/");

    m_file_stream_
    << std::endl
    << "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2\">\n"
    << "<Domain>\n";


}


void XDMFStream::close()
{
    while (!m_path_.empty()) { close_grid(); }

    m_h5_stream_.close();

    m_file_stream_
    << "</Domain>" << std::endl
    << "</Xdmf>" << std::endl;

    m_file_stream_.close();

}


void XDMFStream::open_grid(const std::string &g_name, int TAG)
{
    m_path_.push_back(g_name);

    m_h5_stream_.open_group(path());

    int level = static_cast<int>( m_path_.size());


    switch (TAG)
    {

        case COLLECTION_TEMPORAL:

            m_file_stream_
            << std::endl
            << std::setw(level * 2) << "" << "<Grid Name=\"" << g_name <<
            "\" GridType=\"Collection\" CollectionType=\"Temporal\">" << std::endl;

            break;
        case TREE:

            m_file_stream_
            << std::endl
            << std::setw(level * 2) << "" << "<Grid Name=\"" << g_name <<
            "\" GridType=\"Collection\" CollectionType=\"Tree\">" << std::endl;

            break;
        case UNIFORM://

            m_file_stream_
            << std::endl
            << std::setw(level * 2) << "" << "<Grid Name=\"" << g_name << "\" GridType=\"Uniform\">" << std::endl;
            break;

    }


}


void XDMFStream::time(Real t)
{
    int level = static_cast<int>( m_path_.size());
    m_file_stream_ << std::setw(level * 2) << "" << "   <Time Value=\"" << t << "\"/>" << std::endl;

}

void XDMFStream::close_grid()
{
    if (!m_path_.empty())
    {
        int level = static_cast<int>( m_path_.size());
        m_file_stream_ << std::setw(level * 2) << "" << "</Grid> <!--" << path() << " --> " << std::endl;
        m_path_.pop_back();
    }


}


void _str_replace(std::string *s, std::string const &place_holder, std::string const &txt)
{
    s->replace(s->find(place_holder), place_holder.size(), txt);
}


void XDMFStream::write(std::string const &ds_name, data_model::DataSet const &ds)
{
    if (ds.empty())
    {
        VERBOSE << "Try to write empty dataset: [" << ds_name << "] Ignored!" << std::endl;
        return;
    }

    std::string url;

    url = m_h5_stream_.write(ds_name, ds);

    int level = static_cast<int>( m_path_.size());

    int ndims;

    nTuple<size_t, MAX_NDIMS_OF_ARRAY> dims;

    std::tie(ndims, dims, std::ignore, std::ignore, std::ignore, std::ignore) = ds.data_space.shape();

    if (ds.data_type.is_array())
    {
        int n = ds.data_type.rank();
        for (int i = 0; i < n; ++i)
        {
            dims[ndims + i] = ds.data_type.extent(i);
        }
        ndims += n;
    }

    m_file_stream_ << std::setw(level * 2 + 4) << "" << "<DataItem Dimensions=\"";

    for (int i = 0; i < ndims; ++i) { m_file_stream_ << dims[i] << " "; }

    m_file_stream_ << "\" " << "NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n"

    << std::setw(level * 2 + 6) << "" << url << std::endl
    << std::setw(level * 2 + 4) << "" << "</DataItem>\n";


}

void XDMFStream::write(std::string const &ds_name,
                       base::AttributeObject const &attr)
{
    if (attr.data_set().empty())
    {
        VERBOSE << "Try to write empty Attribute: [" << ds_name << "] Ignored!" << std::endl;

        return;
    }
    static const char a_Center_str[][10] = {
            "Node",
            "Edge",
            "Face",
            "Cell"
    };

    static const char a_AttributeType_str[][10] = {
            "Scalar",
            "Vector",
            "Tensor",
            "Cell"
    };

    int level = static_cast<int>( m_path_.size());

// @FIXME ParaView do not support EDEG or FACE

//    std::string center_type = a_Center_str[attr.center_type()];
//    std::string attr_type = a_AttributeType_str[attr.rank()];

    std::string center_type = "Node";
    std::string attr_type = a_AttributeType_str[
            attr.rank()
            + ((attr.center_type() == 1 || attr.center_type() == 2) ? 1 : 0)
    ];

    m_file_stream_ << ""
    << std::setw(level * 2 + 2) << "" << "<Attribute Name=\"" << ds_name << " \" "
    << "AttributeType=\"" << attr_type << "\" "
    << "Center=\"" << center_type << "\">" << std::endl;

    this->write(ds_name, attr.data_set());

    m_file_stream_ << std::setw(level * 2 + 2) << "" << "</Attribute>" << std::endl;


    VERBOSE << "data_set [" << ds_name << "] is saved in [" << path() << "]!" << std::endl;
}


/**
 *
 *
 *    XML attribute : TopologyType = Polyvertex | Polyline | Polygon |
                                  Triangle | Quadrilateral | Tetrahedron | Pyramid| Wedge | Hexahedron |
                                  Edge_3 | Triangle_6 | Quadrilateral_8 | Tetrahedron_10 | Pyramid_13 |
                                  Wedge_15 | Wedge_18 | Hexahedron_20 | Hexahedron_24 | Hexahedron_27 |
                                  Mixed |
                                  2DSMesh | 2DRectMesh | 2DCoRectMesh |
                                  3DSMesh | 3DRectMesh | 3DCoRectMesh
 */

void XDMFStream::set_topology_geometry(std::string const &name, int ndims, size_t const *dims, Real const *xmin,
                                       Real const *dx)
{

    int level = static_cast<int>(m_path_.size());

    if (ndims == 3)
    {
        m_file_stream_ << ""
        << std::setw(level * 2 + 2) << "" << "<Topology TopologyType=\"3DCoRectMesh\""
        << std::setw(level * 2 + 2) << "" << " Dimensions=\"" << dims[0] << " " << dims[1] << " " << dims[2] << "\"/>\n"
        << std::setw(level * 2 + 2) << "" << "<geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
        << std::setw(level * 2 + 2) << "" << "  <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" "
                " Precision=\"4\" Format=\"XML\">" << xmin[0] << " " << xmin[1] << " " << xmin[2] << "</DataItem>\n"
        << std::setw(level * 2 + 2) << "" << "  <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\""
                " Precision=\"4\" Format=\"XML\">" << dx[0] << " " << dx[1] << " " << dx[2] << "  </DataItem >\n"
        << std::setw(level * 2 + 2) << "" << "</geometry>";


    }
    else if (ndims == 2)
    {
        m_file_stream_ << ""
        << std::setw(level * 2 + 2) << "" << "<Topology TopologyType=\"2DCoRectMesh\""
        << std::setw(level * 2 + 2) << "" << "      Dimensions=\"" << dims[0] << " " << dims[1] << "\"/>\n"
        << std::setw(level * 2 + 2) << "" << "<geometry GeometryType=\"ORIGIN_DXDY\">\n"
        << std::setw(level * 2 + 2) << "" << "   <DataItem Name=\"Origin\" Dimensions=\"2\" NumberType=\"Float\" "
                " Precision=\"4\" Format=\"XML\">" << xmin[0] << " " << xmin[1] << "</DataItem>\n"
        << std::setw(level * 2 + 2) << "" << "   <DataItem Name=\"Spacing\" Dimensions=\"2\" NumberType=\"Float\""
                " Precision=\"4\" Format=\"XML\">" << dx[0] << " " << dx[1] << "  </DataItem >\n"
        << std::setw(level * 2 + 2) << "" << "</geometry>";


    }
    else
    {
        THROW_EXCEPTION_RUNTIME_ERROR(" number of dimension is not 2 or 3");
    }


}


void XDMFStream::set_topology_geometry(std::string const &name, data_model::DataSet const &ds)
{

    int level = static_cast<int>(m_path_.size());

    int ndims;

    nTuple<size_t, MAX_NDIMS_OF_ARRAY> dims;

    std::tie(ndims, dims, std::ignore, std::ignore, std::ignore, std::ignore) = ds.data_space.shape();

    --ndims;
    if (ndims == 2)
    {
        m_file_stream_ << ""
        << std::setw(level * 2 + 2) << "<Topology Name=\"" << name << "\" "
        << " TopologyType=\"2DSMesh\""
        << " NumberOfElements=\"" << dims[0] << " " << dims[1] << " " << dims[2] << "\"/>\n"
        << std::setw(level * 2 + 2) << "<geometry Name=  \"" << name << "\" GeometryType=\"XY\">" << std::endl;
        this->write("/Grid/points", ds);

        m_file_stream_ << ""
        << std::setw(level * 2 + 2) << "" << " </geometry>\n";
    }

    else if (ndims == 3)
    {
        m_file_stream_ << ""
        << std::setw(level * 2 + 2) << "" << "<Topology Name=\"" << name << "\" "
        << "TopologyType=\"3DSMesh\"  NumberOfElements=\"" << dims[0] << " " << dims[1] << " " << dims[2] << "\"/>\n"
        << std::setw(level * 2 + 2) << "" << "<geometry Name=\"" << name << "\" GeometryType=\"XYZ\">\n";

        this->write("/Grid/points", ds);

        m_file_stream_ << ""
        << std::setw(level * 2 + 2) << "" << "</geometry>\n";
    }
    else
    {
        THROW_EXCEPTION_RUNTIME_ERROR("unsportted grid type");
    }
}

void XDMFStream::reference_topology_geometry(std::string const &id)
{
    int level = static_cast<int>( m_path_.size());
    m_file_stream_
    << std::setw(level * 2) << ""
    << "   <Topology Reference=\"XML\" >/Xdmf/Domain/Topology[@Name=\"" << id << "\"]</Topology>" << std::endl
    << std::setw(level * 2) << ""
    << "   <geometry Reference=\"XML\" >/Xdmf/Domain/geometry[@Name=\"" << id << "\"]</geometry>" << std::endl;

}




//void XDMFStream::set_grid(mesh::MeshPatch const &m)
//{
//    auto b = m.get_box();
//    auto dims = m.get_dimensions();
//    nTuple<Real, 3> dx;
//    dx = (std::get<1>(b) - std::get<0>(b)) / dims;
//    set_grid(3, &dims[0], &std::get<0>(b)[0], &dx[0]);
//
//    for (auto const &item:m.patches())
//    {
//        int level = static_cast<int>(m_path_.size());
//
//        if (item.second->patches().size() > 0)
//        {
//            open_grid(type_cast<std::string>(item.first), item.second->time(), TREE);
//
//            set_grid(*item.second);
//
//            close_grid();
//        }
//    }
//
//
//}
}}