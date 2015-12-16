/**
 * @file xdmf_stream.cpp.cpp
 * @author salmon
 * @date 2015-12-02.
 */
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include "xdmf_stream.h"

#include "io.h"
#include "../gtl/utilities/log.h"
#include "../data_model/dataset.h"
#include "../parallel/parallel.h"

namespace simpla { namespace io
{
struct XDMFStream::pimpl_s
{


    typedef parallel::concurrent_hash_map<std::string,
            std::tuple<int, std::shared_ptr<const DataSet> >> container_type;

    container_type m_datasets_;

    std::list<std::string> m_grid_name_;
    /**
     *  <-1 => grid is not set
     *  =-1 => record is not started
     *  >=0 => record is started,
     */
    int m_record_count_;

    std::string m_prefix_;
    std::ofstream m_file_stream_;


};


XDMFStream::XDMFStream() : m_pimpl_(new pimpl_s)
{
    m_pimpl_->m_record_count_ = -2;
    m_pimpl_->m_prefix_ = "";
}

XDMFStream::~XDMFStream()
{
    close();
}

std::string XDMFStream::path() const
{
    std::ostringstream buffer;
    buffer << "/";
    for (auto const &g_name:m_pimpl_->m_grid_name_)
    {
        buffer << g_name << "/";
    }
    return buffer.str();
}

void XDMFStream::open(std::string const &prefix, std::string const &grid_name)
{

    m_pimpl_->m_prefix_ = prefix;


    close();

    io::cd(m_pimpl_->m_prefix_ + ".h5:/");

    m_pimpl_->m_file_stream_.open(m_pimpl_->m_prefix_ + ".xdmf");
    m_pimpl_->m_file_stream_ << ""
            "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2\">\n"
            "<Domain>\n";

}

void XDMFStream::close()
{
    stop_record();

    while (!m_pimpl_->m_grid_name_.empty()) { close_grid(); }

    if (m_pimpl_->m_file_stream_.is_open())
    {
        m_pimpl_->m_file_stream_
        << "</Domain>" << std::endl
        << "</Xdmf>" << std::endl;

        m_pimpl_->m_file_stream_.close();
    }
}

void XDMFStream::open_grid(const std::string &g_name, Real time, int TAG)
{

    m_pimpl_->m_grid_name_.push_back(g_name);

    io::cd(path());

    int level = static_cast<int>(m_pimpl_->m_grid_name_.size());


    switch (TAG)
    {

        case COLLECTION_TEMPORAL:

            m_pimpl_->m_file_stream_ << std::endl

            << std::setw(level * 2) << "" << "<Grid Name=\"" << g_name <<
            "\" GridType=\"Collection\" CollectionType=\"Temporal\">" << std::endl;

            break;
        case TREE:

            m_pimpl_->m_file_stream_ << std::endl

            << std::setw(level * 2) << "" << "<Grid Name=\"" << g_name <<
            "\" GridType=\"Collection\" CollectionType=\"Tree\">" << std::endl;

            break;
        default://

            m_pimpl_->m_file_stream_
            << std::setw(level * 2) << "" << "<Grid Name=\"" << g_name << "\" GridType=\"Uniform\">" << std::endl
            << std::setw(level * 2) << "" << "   <Topology Reference=\"/Xdmf/Domain/Topology[1]\"/>" << std::endl
            << std::setw(level * 2) << "" << "   <Geometry Reference=\"/Xdmf/Domain/Geometry[1]\"/>" << std::endl
            << std::setw(level * 2) << "" << "   <Time Value=\"" << time << "\"/>" << std::endl;


            break;

    }

}

void XDMFStream::close_grid()
{
    if (!m_pimpl_->m_grid_name_.empty())
    {
        int level = static_cast<int>(m_pimpl_->m_grid_name_.size());

        m_pimpl_->m_file_stream_ << std::setw(level * 2) << "" << "</Grid> <!--" << path() << " --> " << std::endl;

        m_pimpl_->m_grid_name_.pop_back();
    }


}

void XDMFStream::start_record(std::string const &s)
{
    std::string g_name = s;

    if (s == "") { g_name = "unanmed"; }


    if (m_pimpl_->m_record_count_ < 0)
    {
        open_grid(g_name, 0, COLLECTION_TEMPORAL);

        m_pimpl_->m_record_count_ = 0;
    }


}

void XDMFStream::record(Real t)
{
    if (m_pimpl_->m_record_count_ < 0) { start_record(); }

    write(0);

    ++m_pimpl_->m_record_count_;
}

void XDMFStream::stop_record()
{
    if (m_pimpl_->m_record_count_ >= 0)
    {
        close_grid();
        m_pimpl_->m_record_count_ = 0;
    }
}


void _str_replace(std::string *s, std::string const &place_holder, std::string const &txt)
{
    s->replace(s->find(place_holder), place_holder.size(), txt);
}


void XDMFStream::write_dataitem(std::string const &ds_name, DataSet const &ds)
{

    std::string url = io::save(ds_name, ds);

    int level = static_cast<int>(m_pimpl_->m_grid_name_.size());

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

    m_pimpl_->m_file_stream_

    << std::setw(level * 2 + 4) << "" << "<DataItem Dimensions=\"";

    for (int i = 0; i < ndims; ++i) { m_pimpl_->m_file_stream_ << dims[i] << " "; }

    m_pimpl_->m_file_stream_
    << "\" " << "NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n"

    << std::setw(level * 2 + 6) << "" << url << std::endl
    << std::setw(level * 2 + 4) << "" << "</DataItem>\n";
}

//template<typename T>
//std::string save_dataset(std::string const &prefix, std::string const ds_name, size_t num, T const *p)
//{
//    //TODO xdmf datatype convert
//    std::string url = io::save(ds_name, num, p);
//
//    VERBOSE << "write data item [" << url << "/" << "]" << std::endl;
//
//    std::ostringstream buffer;
//
//    buffer
//    << "      <DataItem Dimensions=\"" << num << "\" " << "NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n"
//    << url << std::endl
//    << "      </DataItem>";
//
//    return buffer.str();
//}

void XDMFStream::write_attribute(std::string const &ds_name, DataSet const &ds, int tag)
{
    static const char a_center_str[][10] = {
            "Node",
            "Edge",
            "Face",
            "Cell"
    };

    int level = static_cast<int>(m_pimpl_->m_grid_name_.size());

    m_pimpl_->m_file_stream_ << ""
    << std::setw(level * 2 + 2) << "" << "<Attribute Name=\"" << ds_name << "\"  AttributeType=\""
    << (((tag & 0xF0) == 0) ? "Scalar" : "Vector")

    << "\" Center=\"" << /* a_center_str[tag&(0x0F)] */ "Node" << "\">\n";  // NOTE paraview only support "Node" element

    write_dataitem(ds_name, ds);

    m_pimpl_->m_file_stream_
    << std::setw(level * 2 + 2) << "" << "</Attribute>" << std::endl;


    VERBOSE << "DataSet [" << ds_name << "] is saved in [" << path() << "]!" << std::endl;
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

    int level = static_cast<int>(m_pimpl_->m_grid_name_.size());

    if (ndims == 3)
    {
        m_pimpl_->m_file_stream_ << ""
        << std::setw(level * 2 + 2) << "" << "<Topology TopologyType=\"3DCoRectMesh\""
        << std::setw(level * 2 + 2) << "" << " Dimensions=\"" << dims[0] << " " << dims[1] << " " << dims[2] << "\"/>\n"
        << std::setw(level * 2 + 2) << "" << "<Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
        << std::setw(level * 2 + 2) << "" << "  <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" "
                " Precision=\"4\" Format=\"XML\">" << xmin[0] << " " << xmin[1] << " " << xmin[2] << "</DataItem>\n"
        << std::setw(level * 2 + 2) << "" << "  <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\""
                " Precision=\"4\" Format=\"XML\">" << dx[0] << " " << dx[1] << " " << dx[2] << "  </DataItem >\n"
        << std::setw(level * 2 + 2) << "" << "</Geometry>";


    }
    else if (ndims == 2)
    {
        m_pimpl_->m_file_stream_ << ""
        << std::setw(level * 2 + 2) << "" << "<Topology TopologyType=\"2DCoRectMesh\""
        << std::setw(level * 2 + 2) << "" << "      Dimensions=\"" << dims[0] << " " << dims[1] << "\"/>\n"
        << std::setw(level * 2 + 2) << "" << "<Geometry GeometryType=\"ORIGIN_DXDY\">\n"
        << std::setw(level * 2 + 2) << "" << "   <DataItem Name=\"Origin\" Dimensions=\"2\" NumberType=\"Float\" "
                " Precision=\"4\" Format=\"XML\">" << xmin[0] << " " << xmin[1] << "</DataItem>\n"
        << std::setw(level * 2 + 2) << "" << "   <DataItem Name=\"Spacing\" Dimensions=\"2\" NumberType=\"Float\""
                " Precision=\"4\" Format=\"XML\">" << dx[0] << " " << dx[1] << "  </DataItem >\n"
        << std::setw(level * 2 + 2) << "" << "</Geometry>";


    }
    else
    {
        THROW_EXCEPTION_RUNTIME_ERROR(" number of dimension is not 2 or 3");
    }


}


void  XDMFStream::set_grid(DataSet const &ds)
{

    int level = static_cast<int>(m_pimpl_->m_grid_name_.size());

    int ndims;

    nTuple<size_t, MAX_NDIMS_OF_ARRAY> dims;

    std::tie(ndims, dims, std::ignore, std::ignore, std::ignore, std::ignore) = ds.dataspace.shape();

    --ndims;
    if (ndims == 2)
    {
        m_pimpl_->m_file_stream_ << ""
        << std::setw(level * 2 + 2) << "" << "<Topology TopologyType=\"2DSMesh\""
        << "       NumberOfElements=\"" << dims[0] << " " << dims[1] << " " << dims[2] << "\"/>\n"
        << std::setw(level * 2 + 2) << "" << "<Geometry GeometryType=\"XY\">\n";

        write_dataitem("/Grid/points", ds);

        m_pimpl_->m_file_stream_ << ""
        << std::setw(level * 2 + 2) << "" << " </Geometry>\n";
    }

    else if (ndims == 3)
    {
        m_pimpl_->m_file_stream_ << ""
        << std::setw(level * 2 + 2) << "" << "<Topology TopologyType=\"3DSMesh\""
        << "       NumberOfElements=\"" << dims[0] << " " << dims[1] << " " << dims[2] << "\"/>\n"
        << std::setw(level * 2 + 2) << "" << "<Geometry GeometryType=\"XYZ\">\n";

        write_dataitem("/Grid/points", ds);

        m_pimpl_->m_file_stream_ << ""
        << std::setw(level * 2 + 2) << "" << "</Geometry>\n";
    }
    else
    {
        THROW_EXCEPTION_RUNTIME_ERROR("unsportted grid type");
    }
}


void XDMFStream::enroll(std::string const &ds_name, DataSet const &ds, int tag)
{
    typename pimpl_s::container_type::accessor acc;

    if (!m_pimpl_->m_datasets_.insert(acc, ds_name))
    {
        THROW_EXCEPTION_RUNTIME_ERROR("DataSet [" + ds_name + "] is registered!");
    }
    else
    {
//        std::get<0>(acc->second) = tag;
//        std::get<1>(acc->second) = ds.shared_from_this();
//
//        VERBOSE << "DataSet [" << ds_name << "] is enrolled to [" << path() << "]!" << std::endl;
    }


}

void XDMFStream::read()
{
    UNIMPLEMENTED;
}


void  XDMFStream::write(Real t)
{

    std::string g_name = type_cast<std::string>(m_pimpl_->m_record_count_);

    open_grid(g_name, 0, t);

    io::cd(path());

    for (auto const &item:    m_pimpl_->m_datasets_)
    {
        write_attribute(item.first, *std::get<1>(item.second), std::get<0>(item.second));
    }

    close_grid();
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
//        int level = static_cast<int>(m_pimpl_->m_grid_name_.size());
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