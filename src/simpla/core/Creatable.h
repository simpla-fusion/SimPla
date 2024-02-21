//
// Created by salmon on 17-11-20.
//

#ifndef SIMPLA_CREATABLE_H
#define SIMPLA_CREATABLE_H

#include <simpla/utilities/Factory.h>
#include <memory>
#include "DataEntry.h"
#include "Serializable.h"
namespace simpla {
namespace data {

template <typename TObj>
struct Creatable {
   public:
    //    template <typename... Args>
    //    static std::shared_ptr<TObj> Create(Args &&... args) {
    //        return std::shared_ptr<TObj>(new TObj(std::forward<Args>(args)...));
    //    }
    static std::shared_ptr<TObj> Create(std::string const &k) { return simpla::Factory<TObj>::Create(k); }
    static std::shared_ptr<TObj> Create(std::shared_ptr<const DataEntry> const &k) {
        auto res = simpla::Factory<TObj>::Create(k->GetValue<std::string>("_REGISTER_NAME_", ""));
        if (auto s = std::dynamic_pointer_cast<Serializable>(res)) { s->Deserialize(k); }
        return res;
    }
    template <typename U, typename... Args>
    static std::shared_ptr<U> CreateAs(Args &&... args) {
        return std::dynamic_pointer_cast<U>(Create(std::forward<Args>(args)...));
    }
};
}  // namespace data {
}  // namespace simpla {

#endif  // SIMPLA_CREATABLE_H
