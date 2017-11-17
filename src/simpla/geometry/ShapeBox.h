//
// Created by salmon on 17-11-14.
//

#ifndef SIMPLA_PARTSHAPE_H
#define SIMPLA_PARTSHAPE_H

#include <simpla/utilities/SPDefines.h>
#include "GeoObject.h"

namespace simpla {
namespace geometry {
struct ShapeBoxBase {
   protected:
    explicit ShapeBoxBase(box_type const &b) : m_parameter_box_(b) {}

   public:
    void Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg);
    std::shared_ptr<simpla::data::DataEntry> Serialize() const;
    box_type GetParameterBox() const { return m_parameter_box_; }
    void SetParameterBox(box_type const &b) { m_parameter_box_ = b; }
    virtual std::shared_ptr<const GeoObject> GetBaseShape() const = 0;
    box_type m_parameter_box_{{0, 0, 0}, {1, 1, 1}};
};
template <typename BaseShape>
struct ShapeBox : public BaseShape, public ShapeBoxBase {
    SP_GEO_OBJECT_HEAD(ShapeBox, BaseShape)
   protected:
    template <typename... Args>
    explicit ShapeBox(box_type const &b, Args &&... args) : BaseShape(std::forward<Args>(args)...), ShapeBoxBase(b) {}

   public:
    std::shared_ptr<const GeoObject> GetBaseShape() const override { return BaseShape::shared_from_this(); };
};
template <typename BaseShape>
ShapeBox<BaseShape>::ShapeBox() = default;
template <typename BaseShape>
ShapeBox<BaseShape>::ShapeBox(ShapeBox const &) = default;
template <typename BaseShape>
ShapeBox<BaseShape>::~ShapeBox() = default;
template <typename BaseShape>
ShapeBox<BaseShape>::ShapeBox(Axis const &axis) : base_type(axis){};

template <typename BaseShape>
void ShapeBox<BaseShape>::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    ShapeBoxBase::Deserialize(cfg->Get("Parameter"));
};
template <typename BaseShape>
std::shared_ptr<simpla::data::DataEntry> ShapeBox<BaseShape>::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Parameter", ShapeBoxBase::Serialize());
    return res;
};
}  // namespace geometry {
}  // namespace simpla {

#endif  // SIMPLA_PARTSHAPE_H
