//
// Created by salmon on 17-11-14.
//

#include "spRectangle.h"
namespace simpla {
namespace geometry {
spRectangle::spRectangle() = default;
spRectangle::~spRectangle() = default;
spRectangle::spRectangle(spRectangle const &other) = default;
spRectangle::spRectangle(Real l, Real w) : Face(), m_l_(l), m_w_(w) {}

spRectangle::spRectangle(Axis const &axis) : Face(axis) {}
void spRectangle::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) {
    m_w_ = cfg->GetValue("Width", m_w_);
    m_l_ = cfg->GetValue("Length", m_l_);
};
std::shared_ptr<simpla::data::DataEntry> spRectangle::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Width", m_w_);
    res->SetValue("Length", m_l_);
    return res;
};
}  // namespace geometry
}  // namespace simpla