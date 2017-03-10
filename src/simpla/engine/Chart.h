//
// Created by salmon on 17-2-17.
//

#ifndef SIMPLA_CHART_H
#define SIMPLA_CHART_H

#include <memory>
namespace simpla {
namespace engine {

class Chart {
    Chart(int level = 0);
    virtual ~Chart();
    void Connect(Chart *);
    virtual void Disconnect(Chart *p = nullptr);
    int level() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine{
}  // namespace simpla{
#endif  // SIMPLA_CHART_H
