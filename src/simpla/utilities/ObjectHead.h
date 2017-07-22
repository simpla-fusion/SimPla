//
// Created by salmon on 17-7-8.
//

#ifndef SIMPLA_SPOBJECTHEAD_H
#define SIMPLA_SPOBJECTHEAD_H

#include <typeinfo>

namespace simpla {
#define SP_DECLARE_NAME(_CLASS_NAME_)                                \
    virtual std::string GetClassName() const { return ClassName(); } \
    static std::string ClassName() { return __STRING(_CLASS_NAME_); }

#define SP_DEFAULT_CONSTRUCT(_CLASS_NAME_)                 \
    _CLASS_NAME_(this_type const &other) = delete;         \
    _CLASS_NAME_(this_type &&other) = delete;              \
    this_type &operator=(this_type const &other) = delete; \
    this_type &operator=(this_type &&other) = delete;

#define SP_OBJECT_BASE(_BASE_CLASS_NAME_)                                                            \
   private:                                                                                          \
    typedef _BASE_CLASS_NAME_ this_type;                                                             \
                                                                                                     \
   public:                                                                                           \
    virtual bool isA(const std::type_info &info) const { return typeid(_BASE_CLASS_NAME_) == info; } \
    template <typename _UOTHER_>                                                                     \
    bool isA() const {                                                                               \
        return isA(typeid(_UOTHER_));                                                                \
    }                                                                                                \
    template <typename U_>                                                                           \
    U_ &cast_as() {                                                                                  \
        return *dynamic_cast<U_ *>(this);                                                            \
    }                                                                                                \
    template <typename U_>                                                                           \
    U_ const &cast_as() const {                                                                      \
        return *dynamic_cast<U_ const *>(this);                                                      \
    }                                                                                                \
    virtual std::type_info const &GetTypeInfo() const { return typeid(_BASE_CLASS_NAME_); }          \
    static std::string GetFancyTypeName_s() { return __STRING(_BASE_CLASS_NAME_); }                  \
    virtual std::string GetFancyTypeName() const { return GetFancyTypeName_s(); }

/**
 * @brief define the common part of the derived class
 */
#define SP_OBJECT_HEAD(_CLASS_NAME_, _BASE_CLASS_NAME_)                                    \
   public:                                                                                 \
    bool isA(std::type_info const &info) const override {                                  \
        return typeid(_CLASS_NAME_) == info || _BASE_CLASS_NAME_::isA(info);               \
    }                                                                                      \
    std::type_info const &GetTypeInfo() const override { return typeid(_CLASS_NAME_); }    \
    static std::string GetFancyTypeName_s() { return __STRING(_CLASS_NAME_); }             \
    virtual std::string GetFancyTypeName() const override { return GetFancyTypeName_s(); } \
    static bool _is_registered;                                                            \
                                                                                           \
   private:                                                                                \
    typedef _BASE_CLASS_NAME_ base_type;                                                   \
    typedef _CLASS_NAME_ this_type;                                                        \
                                                                                           \
   public:
}
#endif  // SIMPLA_SPOBJECTHEAD_H
