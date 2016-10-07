/**
 * @file distributed_object.h
 * @author salmon
 * @date 2015-10-17.
 */

#ifndef SIMPLA_DISTRIBUTED_OBJECT_H
#define SIMPLA_DISTRIBUTED_OBJECT_H

#include <vector>
#include <memory>
#include <boost/uuid/uuid.hpp>
#include "../sp_def.h"

namespace simpla { namespace toolbox
{
class DataSet;

struct DistributedObject
{
    DistributedObject();

    DistributedObject(DistributedObject const &) = delete;

    ~DistributedObject();

    void clear();

    void sync();

    void wait();

    void add_send_link(size_t id, const nTuple<int, 3> &offset, const DataSet *);

    void add_recv_link(size_t id, const nTuple<int, 3> &offset, DataSet *);

    bool is_ready() const;


private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> pimpl_;

};
}}//namespace simpla
#endif //SIMPLA_DISTRIBUTED_OBJECT_H
