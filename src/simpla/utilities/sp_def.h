//
// Created by salmon on 16-6-27.
//

#ifndef SIMPLA_SP_DEF_H
#define SIMPLA_SP_DEF_H

#include <simpla/SIMPLA_config.h>
#include <cassert>
#include <limits>
#include <string>
namespace simpla {

// enum POSITION
//{
//	/*
//	 FULL = -1, // 11111111
//	 CENTER = 0, // 00000000
//	 LEFT = 1, // 00000001
//	 RIGHT = 2, // 00000010
//	 DOWN = 4, // 00000100
//	 UP = 8, // 00001000
//	 BACK = 16, // 00010000
//	 FRONT = 32 //00100000
//	 */
//	FULL = -1, //!< FULL
//	CENTER = 0, //!< CENTER
//	LEFT = 1,  //!< LEFT
//	RIGHT = 2, //!< RIGHT
//	DOWN = 4,  //!< DOWN
//	UP = 8,    //!< UP
//	BACK = 16, //!< BACK
//	FRONT = 32 //!< FRONT
//};
//
enum ArrayOrder {
    C_ORDER,       // SLOW FIRST
    FORTRAN_ORDER  //  FAST_FIRST
};

typedef Real scalar_type;

static constexpr Real INIFITY = std::numeric_limits<Real>::infinity();

static constexpr Real EPSILON = std::numeric_limits<Real>::epsilon();
}

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
        assert(isA(typeid(U_)));                                                                     \
        return *dynamic_cast<U_ *>(this);                                                            \
    }                                                                                                \
    template <typename U_>                                                                           \
    U_ const &cast_as() const {                                                                      \
        assert(isA(typeid(U_)));                                                                     \
        return *dynamic_cast<U_ const *>(this);                                                      \
    }                                                                                                \
    virtual std::type_info const &GetTypeInfo() const { return typeid(_BASE_CLASS_NAME_); }          \
    virtual std::string GetClassName() const { return __STRING(_BASE_CLASS_NAME_); }                 \
    static std::string ClassName() { return __STRING(_BASE_CLASS_NAME_); }

/**
 * @brief define the common part of the derived class
 */
#define SP_OBJECT_HEAD(_CLASS_NAME_, _BASE_CLASS_NAME_)                                 \
   public:                                                                              \
    bool isA(std::type_info const &info) const override {                               \
        return typeid(_CLASS_NAME_) == info || _BASE_CLASS_NAME_::isA(info);            \
    }                                                                                   \
    std::type_info const &GetTypeInfo() const override { return typeid(_CLASS_NAME_); } \
    std::string GetClassName() const override { return __STRING(_CLASS_NAME_); }        \
    static std::string ClassName() { return __STRING(_CLASS_NAME_); }                   \
                                                                                        \
   private:                                                                             \
    typedef _BASE_CLASS_NAME_ base_type;                                                \
    typedef _CLASS_NAME_ this_type;                                                     \
                                                                                        \
   public:

#define SP_DEFAULT_CONSTRUCT(_CLASS_NAME_)                       \
    _CLASS_NAME_(this_type const &other) = delete;            \
    _CLASS_NAME_(this_type &&other) = delete;                 \
    this_type &operator=(this_type const &other) = delete; \
    this_type &operator=(this_type &&other) = delete;

#define ENABLE_IF(_COND_) std::enable_if_t<_COND_, void> *_p = nullptr

#endif  // SIMPLA_SP_DEF_H
