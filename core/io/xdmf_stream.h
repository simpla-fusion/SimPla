/**
 * @file xdmf_stream.h
 * @author salmon
 * @date 2015-12-02.
 */

#ifndef SIMPLA_XDMF_STREAM_H
#define SIMPLA_XDMF_STREAM_H

#include <memory>
#include "../dataset/dataset.h"

namespace simpla { template<typename ...> class Field; }

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

    void write(Real t);


    enum
    {
        TAG_NODE = 0, TAG_EDGE = 1, TAG_FACE = 2, TAG_CELL = 3
    };

    enum
    {
        UNIFORM = 0, COLLECTION_TEMPORAL = 1
    };


    void write_dataitem(std::string const &s, DataSet const &ds);

    void write_attribute(std::string const &s, DataSet const &ds, int tag = TAG_NODE);

    void start_record(std::string const &s = "");

    void record(Real t);

    void stop_record();

    void open_grid(const std::string &g_name, Real t, int TAG);

    void close_grid();

    void enroll(std::string const &, DataSet const &ds, int tag = TAG_NODE);


    void set_grid(int ndims, size_t const *dims, Real const *xmin, Real const *dx);

    void set_grid(DataSet const &ds);

    bool check_grid() const;

    std::string path() const;

    template<typename TV, typename TM, int IFORM>
    void enroll(std::string const &name,
                Field<TV, TM, std::integral_constant<int, IFORM>> const &f)
    {
        enroll(name, f.dataset(),
               IFORM | ((traits::is_ntuple<TV>::value
                         || (IFORM == 1 || IFORM == 2)) ? 0x10 : 0));

    };

    template<typename TV, typename TM, int IFORM>
    void write_attribute(std::string const &s,
                         Field<TV, TM, std::integral_constant<int, IFORM>> const &f)
    {
        write_attribute(s, f.dataset(),
                        IFORM |
                        ((traits::is_ntuple<TV>::value
                          || (IFORM == 1 || IFORM == 2)) ? 0x10 : 0));
    }


private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};


}}//namespace simpla{namespace io{

#endif //SIMPLA_XDMF_STREAM_H
