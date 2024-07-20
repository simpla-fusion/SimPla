#ifndef SIMPLA_DOMAIN_HPP
#define SIMPLA_DOMAIN_HPP

#include "simpla/core/data_block.hpp"

namespace simpla {

class DomainBase : public std::enable_shared_from_this<DomainBase> {};

template <int NDIM = 1>
class Domain : public DomainBase {
    typedef Domain<NDIM> ThisType;

   public:
    template <typename TValue, int IFORM, typename... Args>
    DataBlock<TValue> create_datablock(Args &&...args) const {
        return DataBlock<TValue>(std::forward<Args>(args)...);
    }

    template <typename TValue, typename... Args>
    void set_value(DataBlock<TValue> &data, Args &&...args) const {
        return data.set_value(std::forward<Args>(args)...);
    }

    template <typename TValue, typename... Args>
    void get_value(DataBlock<TValue> const &data, Args &&...args) const {
        return data.get_value(std::forward<Args>(args)...);
    }
};

};  // namespace simpla

#endif  // SIMPLA_DOMAIN_HPP