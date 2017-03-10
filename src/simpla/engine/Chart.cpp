//
// Created by salmon on 17-2-17.
//
#include "Chart.h"
#include <set>
namespace simpla {
namespace engine {
struct Chart::pimpl_s {
    std::set<Chart *> m_connections_;
    const int m_level_ = 0;
    pimpl_s(int l);
    ~pimpl_s();
};
Chart::pimpl_s::pimpl_s(int l) : m_level_(l){};
Chart::pimpl_s::~pimpl_s(){};

Chart::Chart(int level) : m_pimpl_(new pimpl_s(level)) {}
Chart::~Chart() { Disconnect(); }
void Chart::Connect(Chart *other) {
    if (m_pimpl_->m_connections_.find(other) == m_pimpl_->m_connections_.end()) {
        other->Connect(this);
        m_pimpl_->m_connections_.insert(other);
    }
}
void Chart::Disconnect(Chart *other) {
    if (m_pimpl_->m_connections_.find(other) != m_pimpl_->m_connections_.end()) {
        other->m_pimpl_->m_connections_.erase(this);
        other->Disconnect(this);
    }
}
int Chart::level() const { return m_pimpl_->m_level_; }
}  // namespace engine{
}  // namespace simpla{
