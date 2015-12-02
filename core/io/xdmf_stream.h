/**
 * @file xdmf_stream.h
 * @author salmon
 * @date 2015-12-02.
 */

#ifndef SIMPLA_XDMF_STREAM_H
#define SIMPLA_XDMF_STREAM_H

#include <memory>
#include "../dataset/dataset.h"

namespace simpla { namespace io
{

class XDMFStream
{
public:

    XDMFStream();

    virtual  ~XDMFStream();

    void set_grid(int ndims, size_t const *dims, Real const *xmin, Real const *dx);

    void set_grid(DataSet const &ds);

    void open(std::string const &, std::string const &grid_name = "Unamed");


    void close();

    void read();

    void write();

    virtual void next_time_step();

    virtual void set_grid();

    virtual Real time() const = 0;

    virtual DataSet grid_vertices() const = 0;

    enum
    {
        TAG_NODE = 0, TAG_EDGE = 1, TAG_FACE = 2, TAG_CELL = 3
    };

    void register_dataset(std::string const &, DataSet const &ds, int tag = TAG_NODE);

    template<typename T>
    void register_dataset(std::string const &name, T &obj, int tag = TAG_NODE)
    {
        register_dataset(name, traits::make_dataset(obj), tag);
    };

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};

}}//namespace simpla{namespace io{

#endif //SIMPLA_XDMF_STREAM_H
