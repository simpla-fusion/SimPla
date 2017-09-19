//
// Created by salmon on 17-9-18.
//

#ifndef SIMPLA_MPIUPDATER_H
#define SIMPLA_MPIUPDATER_H

#include "simpla/SIMPLA_config.h"

#include <simpla/utilities/SPDefines.h>
#include <memory>

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

    void SetGhostWidth(index_tuple const &);
    index_tuple GetGhostWidth() const;
    void SetIndexBox(index_box_type const &inner);
    index_box_type GetIndexBox() const;
    void SetTag(int tag);

    virtual void SetUp();
    virtual void TearDown();
    bool isSetUp() const;
    bool isEnable() const;

    virtual void Update() const;
    virtual void Push(ArrayBase const &d) = 0;
    virtual void Pop(ArrayBase &d) const = 0;

   protected:
    virtual std::type_info const &value_type_info() const = 0;

    void *GetSendBuffer(int i) const;
    index_box_type GetSendBufferBox(int i) const;
    void *GetRecvBuffer(int i) const;
    index_box_type GetRecvBufferBox(int i) const;

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

    void Push(ArrayBase const &d) override;
    void Pop(ArrayBase &d) const override;

   private:
    Array<V> send_buffer[6];
    Array<V> recv_buffer[6];
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
    if (isSetUp() || !isEnable()) { return; }
    MPIUpdater::SetUp();
    for (int i = 0; i < 6; ++i) {
        send_buffer[i].reset(reinterpret_cast<V *>(GetSendBuffer(i)), GetSendBufferBox(i));
        recv_buffer[i].reset(reinterpret_cast<V *>(GetRecvBuffer(i)), GetRecvBufferBox(i));
        send_buffer[i].Clear();
        recv_buffer[i].Clear();
    }
}
template <typename V>
void MPIUpdaterT<V>::TearDown() {
    for (int i = 0; i < 6; ++i) {
        send_buffer[i].TearDown();
        recv_buffer[i].TearDown();
    }
}
template <typename V>
void MPIUpdaterT<V>::Push(ArrayBase const &d) {
    if (!isSetUp()) { return; }
    for (auto &v : send_buffer) {
        if (!v.empty()) { v.CopyIn(d); }
    }
}
template <typename V>
void MPIUpdaterT<V>::Pop(ArrayBase &d) const {
    if (!isSetUp()) { return; }
    for (auto &v : recv_buffer) {
        if (!v.empty()) { d.CopyIn(v); }
    }
}
}  // namespace parallel
}  // namespace simpla
#endif  // SIMPLA_MPIUPDATER_H
