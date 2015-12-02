/**
 * @file xdmf_stream.cpp.cpp
 * @author salmon
 * @date 2015-12-02.
 */
#include <fstream>
#include <iomanip>
#include <iostream>
#include "xdmf_stream.h"
#include "../gtl/utilities/log.h"
#include "../dataset/dataset.h"
#include "io.h"

namespace simpla { namespace io
{
struct XDMFStream::pimpl_s
{
    pimpl_s();

    ~pimpl_s();

    void open(std::string const &, std::string const &grid_name);

    void close();

    bool read();

    void write();

    void register_dataset(std::string const &url, DataSet const &ds, int IFORM = 0);

    std::map<std::string, std::tuple<int, DataSet>> m_datasets_;

    size_t m_grid_count_;

    std::string m_grid_name_;
    std::string m_prefix_;
    std::ofstream m_file_stream_;

    Real m_time_;

};

XDMFStream::pimpl_s::pimpl_s()
        : m_time_(0),
          m_grid_count_(0),
          m_prefix_(""),
          m_grid_name_("unnamed")
{
}

XDMFStream::pimpl_s::~pimpl_s()
{
    close();
}

void XDMFStream::pimpl_s::open(std::string const &prefix, std::string const &grid_name)
{

    m_prefix_ = prefix;
    m_grid_name_ = grid_name;

    close();

    io::cd(m_prefix_ + ".h5:/");

    m_file_stream_.open(m_prefix_ + ".xdmf");
    m_file_stream_
    << ""
            "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2\">\n"
            "  <Domain>\n";
}

void XDMFStream::set_grid()
{
    m_pimpl_->m_file_stream_
    << "<Grid Name=\"" << m_pimpl_->m_grid_name_ <<
    "\" GridType=\"Collection\" CollectionType=\"Temporal\">" <<
    std::endl;
}

void XDMFStream::pimpl_s::close()
{
    if (m_file_stream_.is_open())
    {
        io::close();
        m_file_stream_
        << "   </Grid> " << std::endl
        << "  </Domain>" << std::endl
        << "</Xdmf>";
        m_file_stream_.close();
    }
}

void _str_replace(std::string *s, std::string const &place_holder, std::string const &txt)
{
    s->replace(s->find(place_holder), place_holder.size(), txt);
}

std::string save_dataset(std::string const ds_name, DataSet const &ds)
{

    std::string url = io::save(ds_name, ds);

    std::ostringstream buffer;

    int ndims;

    nTuple<size_t, MAX_NDIMS_OF_ARRAY> dims;

    std::tie(ndims, dims, std::ignore, std::ignore, std::ignore, std::ignore) = ds.dataspace.shape();

    if (ds.datatype.is_array())
    {
        int n = ds.datatype.rank();
        for (int i = 0; i < n; ++i)
        {
            dims[ndims + i] = ds.datatype.extent(i);
        }
        ndims += n;
    }

    buffer
    << "      <DataItem Dimensions=\"";
    for (int i = 0; i < ndims; ++i)
    {
        buffer << dims[i] << " ";
    }

    buffer << "\" "
    << "NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n"
    << "      " << url << std::endl
    << "      </DataItem>\n";
    return buffer.str();
}

template<typename T>
std::string save_dataset(std::string const &prefix, std::string const ds_name, size_t num, T const *p)
{
    //TODO xdmf datatype convert
    std::string url = io::save(ds_name, num, p);

    VERBOSE << "write data item [" << url << "/" << "]" << std::endl;

    std::ostringstream buffer;

    buffer
    << "      <DataItem Dimensions=\"" << num << "\" " << "NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n"
    << url << std::endl
    << "      </DataItem>";

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

void  XDMFStream::set_grid(int ndims, size_t const *dims, Real const *xmin, Real const *dx)
{

    if (ndims == 3)
    {
        m_pimpl_->m_file_stream_ << ""
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
        m_pimpl_->m_file_stream_ << ""
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


}


void  XDMFStream::set_grid(DataSet const &ds)
{


    int ndims;

    nTuple<size_t, MAX_NDIMS_OF_ARRAY> dims;

    std::tie(ndims, dims, std::ignore, std::ignore, std::ignore, std::ignore) = ds.dataspace.shape();

    --ndims;
    io::cd("/Grid/");
    if (ndims == 2)
    {
        m_pimpl_->m_file_stream_ << ""
        << "  <Topology TopologyType=\"2DSMesh\""
        << "       NumberOfElements=\"" << dims[0] << " " << dims[1] << " " << dims[2] << "\"/>\n"
        << "  <Geometry GeometryType=\"XY\">\n"
        << save_dataset("points", ds)
        << "  </Geometry>\n";
    }

    else if (ndims == 3)
    {
        m_pimpl_->m_file_stream_ << ""
        << "  <Topology TopologyType=\"3DSMesh\""
        << "       NumberOfElements=\"" << dims[0] << " " << dims[1] << " " << dims[2] << "\"/>\n"
        << "  <Geometry GeometryType=\"XYZ\">\n"
        << save_dataset("points", ds)
        << "  </Geometry>\n";
    }
    else
    {
        THROW_EXCEPTION_RUNTIME_ERROR("unsportted grid type");
    }
}


void XDMFStream::pimpl_s::register_dataset(std::string const &ds_name, DataSet const &ds, int tag)
{
    m_datasets_[ds_name] = std::make_tuple(tag, ds);
}

bool XDMFStream::pimpl_s::read()
{
    UNIMPLEMENTED;
    return false;
}


void  XDMFStream::pimpl_s::write()
{
    static const char a_center_str[][10] = {
            "Node",
            "Edge",
            "Face",
            "Cell"
    };

    m_file_stream_
    << " <Grid Name=\"" << m_grid_name_ << m_grid_count_ << "\" GridType=\"Uniform\">" << std::endl
    << "  <Topology Reference=\"/Xdmf/Domain/Topology[1]\"/>" << std::endl
    << "  <Geometry Reference=\"/Xdmf/Domain/Geometry[1]\"/>" << std::endl
    << "  <Time Value=\"" << m_time_ << "\"/>" << std::endl;

    io::cd("/" + m_grid_name_ + type_cast<std::string>(m_grid_count_) + "/");

    int count = 0;
    for (auto const &item: m_datasets_)
    {
        std::string ds_name = item.first;
        int tag = std::get<0>(item.second);
        DataSet const &ds = std::get<1>(item.second);

        m_file_stream_ << ""
        << "    <Attribute Name=\"" << ds_name << "\"  AttributeType=\""
        << ((ds.datatype.is_array() || tag == TAG_EDGE || tag == TAG_FACE) ? "Vector" : "Scalar")

        << "\" Center=\"" << /* a_center_str[tag] */ "Node" << "\">\n"  // NOTE paraview only support "Node" element

        << save_dataset(ds_name, ds)
        << "    </Attribute>" << std::endl;
    }

    m_file_stream_ << " </Grid>" << std::endl;
}

XDMFStream::XDMFStream() : m_pimpl_(new pimpl_s)
{
}

XDMFStream::~XDMFStream()
{

}

void  XDMFStream::write()
{
    m_pimpl_->write();
}


void  XDMFStream::read()
{
    m_pimpl_->read();
}


void XDMFStream::register_dataset(std::string const &name, DataSet const &ds, int TAG)
{
    m_pimpl_->register_dataset(name, ds, TAG);

}


void XDMFStream::next_time_step()
{
    m_pimpl_->m_time_ = time();
    ++m_pimpl_->m_grid_count_;

}

void XDMFStream::open(const std::string &s, const std::string &grid_name)
{
    m_pimpl_->open(s, grid_name);
    set_grid();
}

void XDMFStream::close()
{
    m_pimpl_->close();
}


}}