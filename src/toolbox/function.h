/**
 * @file function.h
 * @author salmon
 * @date 2015-10-25.
 */

#ifndef SIMPLA_FUNCTION_H
#define SIMPLA_FUNCTION_H

namespace simpla
{
template<typename> struct Function;

template<typename TV, typename TFun>
class Function<TV, TFun>
{
public:


    typedef TV result_type;


private:

    typedef Function<value_type, TFun> this_type;

    TFun m_fun_;

public:


    template<typename TF>
    Function(TF const &fun) : m_fun_(fun)
    {
    }

    Function(this_type const &other) :
            m_fun_(other.m_fun_)
    {
    }

    ~Function()
    {
    }


    operator bool() const
    {
        return m_fun_ != nullptr;
    }


    template<typename ...Args>
    result_type operator()(Args &&...args) const
    {
        return static_cast<value_type>(m_fun_(std::forward<Args>(args)...));
    }


}; // class Function
}//namespace simpla

#endif //SIMPLA_FUNCTION_H
