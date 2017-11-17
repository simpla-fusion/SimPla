//
// Created by salmon on 17-11-14.
//

#include "Rectangle.h"
namespace simpla {
namespace geometry {
Rectangle::Rectangle() = default;
Rectangle::~Rectangle() = default;
Rectangle::Rectangle(Rectangle const &other) = default;
Rectangle::Rectangle(Real l, Real w) : Face(), m_l_(l), m_w_(w) {}

Rectangle::Rectangle(Axis const &axis) : Face(axis) {}
void Rectangle::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) {
    m_w_ = cfg->GetValue("Width", m_w_);
    m_l_ = cfg->GetValue("Length", m_l_);
};
std::shared_ptr<simpla::data::DataEntry> Rectangle::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Width", m_w_);
    res->SetValue("Length", m_l_);
    return res;
};
}  // namespace geometry
}  // namespace simpla