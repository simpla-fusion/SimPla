/**
 * @file XDMFStream.h
 * @author salmon
 * @date 2015-12-02.
 */

#ifndef SIMPLA_XDMF_STREAM_H
#define SIMPLA_XDMF_STREAM_H

#include <fstream>
#include <memory>
#include "../data_model/DataSet.h"
#include "../base/Attribute.h"


namespace simpla { namespace io
{

class XDMFStream
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


    std::string path() const;

    void open(std::string const &prefix, std::string const &grid_name = "Unamed");

    void close();

    void open_grid(const std::string &g_name, int TAG);


    void set_topology_geometry(std::string const &name, int ndims, size_t const *dims, Real const *xmin,
                               Real const *dx);

    void set_topology_geometry(std::string const &name, data_model::DataSet const &ds);

    void close_grid();

    void time(Real time);

    void write(std::string const &s, data_model::DataSet const &ds);

    void write(std::string const &s, base::AttributeObject const &ds);

    template<typename AttrList>
    void write(AttrList const &attr_list)
    {
        for (auto const &item:attr_list)
        {
            write(item.first, *std::dynamic_pointer_cast<base::AttributeObject>(item.second.lock()));
        }
    }

    void reference_topology_geometry(std::string const &id);


private:
    std::list<std::string> m_path_;

    std::string m_prefix_;

    std::ofstream m_file_stream_;
};


}}//namespace simpla{namespace io{

#endif //SIMPLA_XDMF_STREAM_H
