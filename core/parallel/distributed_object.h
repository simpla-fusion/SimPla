/**
 * @file distributed_object.h
 * @author salmon
 * @date 2015-10-17.
 */

#ifndef SIMPLA_DISTRIBUTED_OBJECT_H
#define SIMPLA_DISTRIBUTED_OBJECT_H

#include <bits/unique_ptr.h>
#include "mpi_comm.h"
#include "mpi_aux_functions.h"
#include "mpi_update.h"
#include "../dataset/dataset.h"

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
        add(traits::make_dataset(args));
        add(std::forward<Others>(others)...);
    }

    template<typename T>
    void add(T const &args)
    {
        add(traits::make_dataset(args));
    }

    template<typename T>
    void add(T *args)
    {
        add(traits::make_dataset(*args));
    }

    template<typename T>
    void add(T &args)
    {
        add(traits::make_dataset(args));
    }

    void add(DataSet ds);

    inline void add_link_send(nTuple<int, 3> const &coord_offset, DataSet &ds)
    {
        send_buffer.push_back(std::make_tuple(coord_offset, ds));
    };


    void add_link_recv(nTuple<int, 3> const &coord_offset, DataSet &ds)
    {
        recv_buffer.push_back(std::make_tuple(coord_offset, ds));
    };

    template<typename ...Args>
    void add_link_send(nTuple<int, 3> const &coord_offset, Args &&...args)
    {
        send_buffer.push_back(
                std::make_tuple(coord_offset,
                                traits::make_dataset(std::forward<Args>(args)...)));
    };

    template<typename ...Args>
    void add_link_recv(nTuple<int, 3> const &coord_offset, Args &&...args)
    {
        recv_buffer.push_back(
                std::make_tuple(coord_offset,
                                traits::make_dataset(std::forward<Args>(args)...)));
    };


    typedef std::tuple<nTuple<int, 3>, DataSet> link_s;

    std::vector<link_s> send_buffer;
    std::vector<link_s> recv_buffer;
private:


    struct pimpl_s;
    std::unique_ptr<pimpl_s> pimpl_;

};
}}//namespace simpla
#endif //SIMPLA_DISTRIBUTED_OBJECT_H
