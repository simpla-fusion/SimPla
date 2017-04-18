/**
 * @file factory.h
 *
 *  created on: 2014-6-13
 *      Author: salmon
 */

#ifndef FACTORY_H_
#define FACTORY_H_

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/macro.h>
#include <simpla/utilities/Log.h>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
namespace simpla {
namespace design_pattern {
/**
 *  @ingroup design_pattern
 * @addtogroup factory Factory
 * @{
 *  \note  Modern C++ Design, Andrei Alexandrescu , Addison Wesley 2001  Charpt 8
 */
template <typename TKey, typename TRes, typename... Args>
struct Factory : public std::map<TKey, std::function<TRes*(Args...)>> {
    typedef std::function<TRes*(Args...)> create_fun_callback;
    typedef std::map<TKey, create_fun_callback> base_type;

   public:
    Factory() {}
    virtual ~Factory() {}

    TRes* Create(TKey const& id, Args... args) const {
        auto it = this->find(id);
        return (it == this->end()) ? nullptr : it->second(args...);
    }

    template <typename U>
    bool Register(TKey const& k, ENABLE_IF((std::is_base_of<TRes, U>::value))) {
        auto res = this->emplace(k, [&](Args&&... args) -> TRes* { return new U(std::forward<Args>(args)...); }).second;
        if (res) { LOGGER << "Creator [ " << k << " ] is registered!" << std::endl; }

        return res;
    }

    void Unregister(TKey const& k) { this->erase(k); }
};

/** @} */
}  // namespace design_patter{
}  // namespace simpla

#endif /* FACTORY_H_ */
