/**
 * @file distributed_object.h
 * @author salmon
 * @date 2015-10-17.
 */

#ifndef SIMPLA_DISTRIBUTED_OBJECT_H
#define SIMPLA_DISTRIBUTED_OBJECT_H

#include <memory>

namespace simpla
{
namespace data_model { class DataSet; }

namespace parallel
{

struct DistributedObject
{
    DistributedObject();

    DistributedObject(DistributedObject const &) = delete;

    ~DistributedObject();

    void add(int id, data_model::DataSet &ds);

    void sync();

    void wait();

    bool is_ready() const;

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> pimpl_;

};
}
}//namespace simpla
#endif //SIMPLA_DISTRIBUTED_OBJECT_H
