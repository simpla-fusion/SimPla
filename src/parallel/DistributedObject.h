/**
 * @file distributed_object.h
 * @author salmon
 * @date 2015-10-17.
 */

#ifndef SIMPLA_DISTRIBUTED_OBJECT_H
#define SIMPLA_DISTRIBUTED_OBJECT_H

#include <vector>
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

    void clear();

    void sync();

    void wait();

    int add_send_link(int id, const int offset[3], data_model::DataSet);

    int add_recv_link(int id, const int offset[3], data_model::DataSet);

    void add(int id, data_model::DataSet &ds, std::vector<int> *_send_tag = nullptr,
             std::vector<int> *_recv_tag = nullptr);

    void remove(int tag, bool is_recv = false);

    bool is_ready() const;


private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> pimpl_;

};
}
}//namespace simpla
#endif //SIMPLA_DISTRIBUTED_OBJECT_H
