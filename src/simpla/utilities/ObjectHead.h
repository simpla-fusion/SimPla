//
// Created by salmon on 17-7-8.
//

#ifndef SIMPLA_SPOBJECTHEAD_H
#define SIMPLA_SPOBJECTHEAD_H

#include <typeinfo>

#define SP_DECLARE_NAME(_CLASS_NAME_)                                \
    virtual std::string GetClassName() const { return ClassName(); } \
    static std::string ClassName() { return __STRING(_CLASS_NAME_); }

#define SP_DEFAULT_CONSTRUCT(_CLASS_NAME_)                       \
    _CLASS_NAME_(_CLASS_NAME_ const &other) = delete;            \
    _CLASS_NAME_(_CLASS_NAME_ &&other) = delete;                 \
    _CLASS_NAME_ &operator=(_CLASS_NAME_ const &other) = delete; \
    _CLASS_NAME_ &operator=(_CLASS_NAME_ &&other) = delete;

#define SP_OBJECT_BASE(_BASE_CLASS_NAME_)                                           \
   private:                                                                         \
    typedef _BASE_CLASS_NAME_ this_type;                                            \
                                                                                    \
   public:                                                                          \
    static bool _is_registered;                                                     \
                                                                                    \
    static std::string GetFancyTypeName_s() { return __STRING(_BASE_CLASS_NAME_); } \
    virtual std::string GetFancyTypeName() const { return GetFancyTypeName_s(); }

/**
 * @brief define the common part of the derived class
 */
#define SP_OBJECT_HEAD(_CLASS_NAME_, _BASE_CLASS_NAME_)                                    \
   public:                                                                                 \
    static std::string GetFancyTypeName_s() { return __STRING(_CLASS_NAME_); }             \
    virtual std::string GetFancyTypeName() const override { return GetFancyTypeName_s(); } \
    static bool _is_registered;                                                            \
                                                                                           \
   private:                                                                                \
    typedef _BASE_CLASS_NAME_ base_type;                                                   \
    typedef _CLASS_NAME_ this_type;                                                        \
                                                                                           \
   public:

#endif  // SIMPLA_SPOBJECTHEAD_H
