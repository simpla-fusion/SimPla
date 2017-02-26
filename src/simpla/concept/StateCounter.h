//
// Created by salmon on 17-2-22.
//

#ifndef SIMPLA_STATECOUNTER_H
#define SIMPLA_STATECOUNTER_H

#include <simpla/SIMPLA_config.h>
namespace simpla {
namespace concept {
struct StateCounter {
public:
    StateCounter() {}
    virtual ~StateCounter() {}
    virtual bool isModified() const { return m_tag_count_ != m_click_count_; }
    void Click() { ++m_click_count_; }
    size_type GetClickCount() const { return m_click_count_; }
    size_type GetTagCount() const { return m_tag_count_; }
    void Tag() { m_tag_count_ = m_click_count_; }
    void Tag(size_type c) {
        m_tag_count_ = c;
        m_click_count_ = c;
    }

   private:
    size_type m_click_count_ = 0;
    size_type m_tag_count_ = 0;
};

}  // namespace concept

}  // namespace simpla
#endif  // SIMPLA_STATECOUNTER_H
