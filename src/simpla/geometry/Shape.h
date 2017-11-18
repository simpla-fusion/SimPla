//
// Created by salmon on 17-11-6.
//

#ifndef SIMPLA_SHAPE_H
#define SIMPLA_SHAPE_H

#include <memory>
#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct Shape : public data::Serializable, public std::enable_shared_from_this<Shape> {
    SP_SERIALIZABLE_HEAD(data::Serializable, Shape)
    virtual Shape *CopyP() const = 0;
    std::shared_ptr<Shape> Copy() const { return std::shared_ptr<Shape>(CopyP()); }

    //    virtual point_type xyz(Real u, Real v, Real w) const = 0;
    //    virtual point_type uvw(Real x, Real y, Real z) const = 0;
};

#define SP_SHAPE_HEAD(_BASE_NAME_, _CLASS_NAME_, _REGISTER_NAME_)                      \
    SP_SERIALIZABLE_HEAD(_BASE_NAME_, _CLASS_NAME_)                                    \
    static std::string RegisterName() { return __STRING(_REGISTER_NAME_); }            \
                                                                                       \
   private:                                                                            \
    static bool _is_registered;                                                        \
                                                                                       \
   protected:                                                                          \
    _CLASS_NAME_();                                                                    \
    _CLASS_NAME_(_CLASS_NAME_ const &);                                                \
                                                                                       \
   public:                                                                             \
    ~_CLASS_NAME_() override;                                                          \
    template <typename... Args>                                                        \
    static std::shared_ptr<this_type> New(Args &&... args) {                           \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...)); \
    }                                                                                  \
                                                                                       \
    this_type *CopyP() const override { return new this_type(*this); };

//    std::shared_ptr<this_type> CopyThis() const { return std::dynamic_pointer_cast<this_type>(Copy()); };
#define SP_SHAPE_REGISTER(_CLASS_NAME_) \
    bool _CLASS_NAME_::_is_registered = \
        simpla::Factory<Shape>::RegisterCreator<_CLASS_NAME_>(_CLASS_NAME_::RegisterName());
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_SHAPE_H
