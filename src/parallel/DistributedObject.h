/**
 * @file distributed_object.h
 * @author salmon
 * @date 2015-10-17.
 */

#ifndef SIMPLA_DISTRIBUTED_OBJECT_H
#define SIMPLA_DISTRIBUTED_OBJECT_H

#include <bits/unique_ptr.h>
#include "MPIComm.h"
#include "MPIAuxFunctions.h"
#include "MPIUpdate.h"
//#include "../data_model/DataSet.h"
namespace simpla { namespace data_model { class DataSet; }}

namespace simpla { namespace parallel
{


struct DistributedObject
{
    DistributedObject();

    DistributedObject(DistributedObject const &) = delete;

    virtual ~DistributedObject();

    virtual void sync();

    virtual void wait();

    virtual bool is_ready() const;


    template<typename T, typename ...Others>
    void add(T const &args, Others &&...others)
    {
        add(data_model::DataSet::create(args));
        add(std::forward<Others>(others)...);
    }

    template<typename T>
    void add(T const &args)
    {
        add(data_model::DataSet::create(args));
    }

    template<typename T>
    void add(T *args)
    {
        add(data_model::DataSet::create(*args));
    }

    template<typename T>
    void add(T &args)
    {
        add(data_model::DataSet::create(args));
    }

    void add(data_model::DataSet ds);

    inline void add_link_send(nTuple<int, 3> const &coord_offset, data_model::DataSet &ds)
    {
        send_buffer.push_back(std::make_tuple(coord_offset, ds));
    };


    void add_link_recv(nTuple<int, 3> const &coord_offset, data_model::DataSet &ds)
    {
        recv_buffer.push_back(std::make_tuple(coord_offset, ds));
    };

    template<typename ...Args>
    void add_link_send(nTuple<int, 3> const &coord_offset, Args &&...args)
    {
        send_buffer.push_back(
                std::make_tuple(coord_offset,
                                data_model::DataSet::create(std::forward<Args>(args)...)));
    };

    template<typename ...Args>
    void add_link_recv(nTuple<int, 3> const &coord_offset, Args &&...args)
    {
        recv_buffer.push_back(
                std::make_tuple(coord_offset,
                                data_model::DataSet::create(std::forward<Args>(args)...)));
    };


    typedef std::tuple<nTuple<int, 3>, data_model::DataSet> link_s;

    std::vector<link_s> send_buffer;
    std::vector<link_s> recv_buffer;
private:


    struct pimpl_s;
    std::unique_ptr<pimpl_s> pimpl_;

};
}}//namespace simpla
#endif //SIMPLA_DISTRIBUTED_OBJECT_H