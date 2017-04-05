//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_WORKER_H
#define SIMPLA_WORKER_H

#include <memory>
#include "simpla/concept/Configurable.h"
namespace simpla {
namespace engine {

class Schedule;
/**
 * @brief Worker is an abstraction of node, cpu socket, thread or any other compute resource.
 */
class Worker : public concept::Configurable {
   public:
    Worker();
    virtual ~Worker();
    Schedule& GetSchedule() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}
#endif  // SIMPLA_WORKER_H
