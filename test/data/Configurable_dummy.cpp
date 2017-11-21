//
// Created by salmon on 17-11-21.
//
#include <simpla/data/Configurable.h>
#include <simpla/data/Serializable.h>
#include <iostream>
struct ConfigurableDummy : public simpla::data::Configurable, public simpla::data::Serializable {
    SP_PROPERTY(int, A) = 1;
    SP_PROPERTY(double, B) = 2;
};

int main(int argc, char** argv) {
    ConfigurableDummy dummy;
    dummy.SetA(1234455);
    dummy.SetProperty<std::string>("justATest", "hello world!");
    std::cout << "Dummy:" << dummy << std::endl;

    std::cout << " A = " << dummy.GetProperty<int>("A") << " = " << dummy.GetA() << " = " << dummy.m_A_ << std::endl;
    std::cout << " B = " << dummy.GetProperty<double>("B") << " = " << dummy.GetB() << " = " << dummy.m_B_ << std::endl;
    std::cout << " justATest = " << dummy.GetProperty<std::string>("justATest") << std::endl;
}