/**
 * @file XDMFStream.h
 * @author salmon
 * @date 2015-12-02.
 */

#ifndef SIMPLA_XDMF_STREAM_H
#define SIMPLA_XDMF_STREAM_H

#include <fstream>
#include <memory>
#include "DataSet.h"
//#include "../base/Attribute.h"
#include "HDF5Stream.h"


namespace simpla { namespace io
{

class XDMFStream : public IOStream
{
public:

    XDMFStream();

    virtual  ~XDMFStream();

    enum
    {
        TAG_NODE = 0, TAG_EDGE = 1, TAG_FACE = 2, TAG_CELL = 3
    };

    enum
    {
        UNIFORM = 0, COLLECTION_TEMPORAL = 1, TREE = 2
    };

    virtual std::string ext_name() const { return "xmf"; }


    std::string path() const;


    virtual std::tuple<bool, std::string> open(std::string const &url, int flag = 0UL);

    virtual std::string read(std::string const &url, data_model::DataSet *ds, int flag = 0UL) { return ""; };

    void close();

    void open_grid(const std::string &g_name, int TAG);

    virtual void set_attribute(std::string const &url, Properties const &v) { };

    void set_topology_geometry(std::string const &name, int ndims, size_t const *dims, Real const *xmin,
                               Real const *dx);

    void set_topology_geometry(std::string const &name, data_model::DataSet const &ds);

    void close_grid();

    void time(Real time);

    std::string write(std::string const &s, data_model::DataSet const &ds, int flag = 0UL);

//    void write(std::string const &s, base::AttributeObject const &ds);


    void reference_topology_geometry(std::string const &id);

    HDF5Stream &hdf5() { return m_h5_stream_; }

    HDF5Stream const &hdf5() const { return m_h5_stream_; }

private:
    std::list<std::string> m_path_;

    std::string m_prefix_;

    std::ofstream m_file_stream_;

    HDF5Stream m_h5_stream_;
};


}}//namespace simpla{namespace io{

#endif //SIMPLA_XDMF_STREAM_H
