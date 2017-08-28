//
// Created by salmon on 17-7-10.
//

#include <iostream>
#include <memory>

template <typename T>
struct Foo {
    T* m_host_;
    Foo(T* host) : m_host_(host) { host->Register(this); }

    std::string GetName() { return "I'm the Foo"; }
};

template <typename THost>
class FirstPolicy {
    THost* m_self_;

   public:
    explicit FirstPolicy(THost* host) : m_self_(host) {}
    virtual ~FirstPolicy() = default;

    template <typename TOther>
    bool Register(TOther* h) {
        std::cout << "Register:" << h->GetName() << std::endl;
        return true;
    }
    std::string GetName() { return "I'm  the first"; }
};

template <typename THost>
class SecondPolicy {
    THost* m_self_;
    bool m_is_register_;

   public:
    SecondPolicy(THost* host) : m_self_(host), m_is_register_(m_self_->Register(this)) {}
    virtual ~SecondPolicy() = default;

    std::string GetName() { return "I'm the m_node_"; }

    Foo<THost> foo{m_self_};
};

template <template <typename> class... Policies>
class Host : public Policies<Host<Policies...>>... {
   public:
    Host() : Policies<Host<Policies...>>(this)... {}
    ~Host() override = default;

    std::string GetName() { return "I'm the host."; }
};

int main(int argc, char** argv) {
    Host<FirstPolicy, SecondPolicy> foo;

    std::cout << foo.GetName() << std::endl;
}