//
// Created by salmon on 17-3-1.
//

#ifndef SIMPLA_SPOBJECTHEAD_H
#define SIMPLA_SPOBJECTHEAD_H
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
    U_ &as() {                                                                                       \
        ASSERT(isA(typeid(U_)));                                                                     \
        return *static_cast<U_ *>(this);                                                             \
    }                                                                                                \
    template <typename U_>                                                                           \
    U_ const &as() const {                                                                           \
        ASSERT(isA(typeid(U_)));                                                                     \
        return static_cast<U_ const *>(this);                                                        \
    }                                                                                                \
    virtual std::type_index TypeIndex() const { return std::type_index(typeid(_BASE_CLASS_NAME_)); } \
    virtual std::string getClassName() const { return __STRING(_BASE_CLASS_NAME_); }

/**
 * @brief define the common part of the derived class
 */
#define SP_OBJECT_HEAD(_CLASS_NAME_, _BASE_CLASS_NAME_)                                         \
   public:                                                                                      \
    virtual bool isA(std::type_info const &info) const {                                        \
        return typeid(_CLASS_NAME_) == info || _BASE_CLASS_NAME_::isA(info);                    \
    }                                                                                           \
    virtual std::type_index TypeIndex() const { return std::type_index(typeid(_CLASS_NAME_)); } \
    virtual std::string getClassName() const { return __STRING(_CLASS_NAME_); }                 \
                                                                                                \
   private:                                                                                     \
    typedef _BASE_CLASS_NAME_ base_type;                                                        \
    typedef _CLASS_NAME_ this_type;                                                             \
                                                                                                \
   public:

#endif  // SIMPLA_SPOBJECTHEAD_H
