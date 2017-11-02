//
// Created by salmon on 17-9-18.
//

#ifndef SIMPLA_MPIUPDATER_H
#define SIMPLA_MPIUPDATER_H

#include <simpla/utilities/SPDefines.h>
#include <memory>
#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/Array.h"

namespace simpla {
namespace parallel {

struct MPIUpdater {
   protected:
    MPIUpdater();

   public:
    virtual ~MPIUpdater();
    template <typename V>
    static std::shared_ptr<MPIUpdater> New();

    void SetDirection(int);
    int GetDirection() const;
    void SetPeriodic(bool tag = true);
    bool IsPeriodic() const;

    void SetIndexBox(index_box_type const &inner);
    index_box_type GetIndexBox() const;
    void SetHaloWidth(index_tuple const &);
    void SetHaloIndexBox(index_box_type const &b);
    index_box_type GetHaloIndexBox() const;
    void SetTag(int tag);

    virtual void SetUp();
    virtual void Clear();
    virtual void TearDown();
    bool isSetUp() const;

    void Push(ArrayBase const &a);
    void Pop(ArrayBase &a) const;
    void SendRecv();

   protected:
    virtual std::type_info const &value_type_info() const = 0;
    virtual ArrayBase &GetSendBuffer(int i) = 0;
    virtual ArrayBase const &GetSendBuffer(int i) const = 0;
    virtual ArrayBase &GetRecvBuffer(int i) = 0;
    virtual ArrayBase const &GetRecvBuffer(int i) const = 0;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};
template <typename V>
struct MPIUpdaterT : public MPIUpdater {
   protected:
    MPIUpdaterT();

   public:
    ~MPIUpdaterT() override;
    std::type_info const &value_type_info() const override { return typeid(V); };

    static std::shared_ptr<MPIUpdaterT<V>> New();
    void SetUp() override;
    void TearDown() override;

    ArrayBase &GetSendBuffer(int i) override { return send_buffer[i]; }
    ArrayBase const &GetSendBuffer(int i) const override { return send_buffer[i]; }
    ArrayBase &GetRecvBuffer(int i) override { return recv_buffer[i]; }
    ArrayBase const &GetRecvBuffer(int i) const override { return recv_buffer[i]; }

   private:
    Array<V> send_buffer[2];
    Array<V> recv_buffer[2];
};

template <typename V>
std::shared_ptr<MPIUpdater> MPIUpdater::New() {
    return MPIUpdaterT<V>::New();
}

template <typename V>
MPIUpdaterT<V>::MPIUpdaterT() : MPIUpdater() {}
template <typename V>
MPIUpdaterT<V>::~MPIUpdaterT() {}
template <typename V>
std::shared_ptr<MPIUpdaterT<V>> MPIUpdaterT<V>::New() {
    return std::shared_ptr<MPIUpdaterT<V>>(new MPIUpdaterT<V>);
}
template <typename V>
void MPIUpdaterT<V>::SetUp() {
    if (isSetUp()) { return; }
    MPIUpdater::SetUp();
}
template <typename V>
void MPIUpdaterT<V>::TearDown() {
    for (auto &v : send_buffer) { v.free(); }
    for (auto &v : recv_buffer) { v.free(); }
    MPIUpdater::TearDown();
}

}  // namespace parallel
}  // namespace simpla
#endif  // SIMPLA_MPIUPDATER_H
