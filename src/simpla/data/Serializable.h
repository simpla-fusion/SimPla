//
// Created by salmon on 17-11-17.
//

#ifndef SIMPLA_SERIALIZABLE_H
#define SIMPLA_SERIALIZABLE_H

#include "DataEntry.h"
namespace simpla {
namespace data {
struct Serializable {
   public:
    Serializable();
    Serializable(Serializable const &);
    virtual ~Serializable();
    virtual void Deserialize(std::shared_ptr<const DataEntry> const &cfg);
    virtual std::shared_ptr<DataEntry> Serialize() const;

    virtual std::string FancyTypeName() const { return ""; }
    static std::shared_ptr<Serializable> Create(std::string const &cfg);
    static std::shared_ptr<Serializable> Create(std::shared_ptr<const DataEntry> const &cfg);
};
std::ostream &operator<<(std::ostream &os, Serializable const &obj);
std::istream &operator>>(std::istream &is, Serializable &obj);

#define SP_SERIALIZABLE_HEAD(_BASE_NAME_, _CLASS_NAME_)                                                              \
   private:                                                                                                          \
    typedef _CLASS_NAME_ this_type;                                                                                  \
    typedef _BASE_NAME_ base_type;                                                                                   \
                                                                                                                     \
   public:                                                                                                           \
    std::string FancyTypeName() const override { return base_type::FancyTypeName() + "." + __STRING(_CLASS_NAME_); } \
                                                                                                                     \
    static std::shared_ptr<this_type> Create(std::string const &k) {                                                 \
        return std::dynamic_pointer_cast<this_type>(base_type::Create(k));                                           \
    }                                                                                                                \
    static std::shared_ptr<this_type> Create(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {            \
        return std::dynamic_pointer_cast<this_type>(base_type::Create(cfg));                                         \
    }
#define SP_ENABLE_NEW                                                                  \
    template <typename... Args>                                                        \
    static std::shared_ptr<this_type> New(Args &&... args) {                           \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...)); \
    }

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_SERIALIZABLE_H
