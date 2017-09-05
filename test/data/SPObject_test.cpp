//
// Created by salmon on 17-2-13.
//
#include <gtest/gtest.h>

#include <iostream>
#include "simpla/data/Data.h"
#include "simpla/data/SPObject.h"
using namespace simpla;
using namespace simpla::data;
struct DummyObject : public SPObject {
    SP_OBJECT_HEAD(DummyObject, SPObject)
    SP_OBJECT_PROPERTY(Real, Mass);
    SP_OBJECT_PROPERTY(Real, Charge);
};
DummyObject::DummyObject() = default;
DummyObject::~DummyObject() = default;

std::shared_ptr<simpla::data::DataNode> DummyObject::Serialize() const { return base_type::Serialize(); };
void DummyObject::Deserialize(std::shared_ptr<data::DataNode>const & cfg) { base_type::Deserialize(cfg); };
SP_OBJECT_REGISTER(DummyObject)
TEST(SPObject, Dummy) {
    auto objA = DummyObject::New();
    objA->SetMass(1.0);
    objA->SetCharge(-1.0);
    std::cout << *objA->Serialize() << std::endl;

    auto objB = std::dynamic_pointer_cast<DummyObject>(SPObject::New(objA->Serialize()));
    EXPECT_TRUE(objB != nullptr);
    EXPECT_DOUBLE_EQ(objA->GetMass(), objB->GetMass());
    EXPECT_DOUBLE_EQ(objA->GetCharge(), objB->GetCharge());
}
