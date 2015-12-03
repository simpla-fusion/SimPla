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


    void open(std::string const &, std::string const &grid_name = "Unamed");

    void close();

    void read();

    void write();


    enum
    {
        TAG_NODE = 0, TAG_EDGE = 1, TAG_FACE = 2, TAG_CELL = 3
    };

    void enroll(std::string const &, DataSet const &ds, int tag = TAG_NODE);

    virtual Real time() const = 0;

    virtual DataSet grid_vertices() const = 0;

    virtual void set_grid();


    void set_grid(int ndims, size_t const *dims, Real const *xmin, Real const *dx);

    void set_grid(DataSet const &ds);


    virtual void next_time_step();


private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};

}}//namespace simpla{namespace io{

#endif //SIMPLA_XDMF_STREAM_H
