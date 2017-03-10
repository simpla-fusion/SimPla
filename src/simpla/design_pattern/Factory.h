/**
 * @file factory.h
 *
 *  created on: 2014-6-13
 *      Author: salmon
 */

#ifndef FACTORY_H_
#define FACTORY_H_

#include <simpla/SIMPLA_config.h>
#include <simpla/mpl/macro.h>
#include <simpla/toolbox/Log.h>
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
struct Factory : public std::map<TKey, std::function<TRes*(Args&&...)>> {
    typedef TKey key_type;
    typedef std::function<TRes*(Args...)> create_fun_callback;
    typedef std::map<key_type, create_fun_callback> CallbackMap;
    typedef std::map<TKey, TRes*(Args...)> base_type;

   public:
    Factory() {}
    virtual ~Factory() {}

    virtual TRes* Create(key_type const& id, Args... args) const {
        auto it = this->find(id);
        //        if (it == callbacks_.end()) { RUNTIME_ERROR("Can not find id " + value_to_string(id)); }
        return (it == this->end()) ? nullptr : (it->second)(std::forward<Args>(args)...);
    }

    template <typename U>
    bool Register(key_type const& k, ENABLE_IF((std::is_base_of<TRes, U>::value))) {
        return this->emplace(k, [&](Args&&... args) -> TRes* { return new U(std::forward<Args>(args)...); }).second;
    }

    virtual void Unregister(key_type const& k) { this->erase(k); }
};

/** @} */
}  // namespace design_patter{
}  // namespace simpla

#endif /* FACTORY_H_ */
