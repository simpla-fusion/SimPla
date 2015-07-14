//
// Created by salmon on 7/10/15.
//

#ifndef SIMPLA_SIGNAL_H
#define SIMPLA_SIGNAL_H

<<<<<<< HEAD
#include <boost/signals2/signal.hpp>

namespace simpla {

template<typename ...T> using signal= boost::signals2::signal<T...>;

}// namespace simpla
=======

#include <list>
#include <memory>
#include <vector>
#include "../macro.h"


namespace simpla
{


template<typename ...> struct Signal;


/**
 *  @NOTE this is not thread-safe
 */
template<typename TResult, typename ...Args>
struct Signal<TResult(Args...)>
{
	typedef TResult result_type;

	struct Slot;

	typedef typename std::list<Slot>::iterator iterator;

	typedef std::function<result_type(Args...)> callback_type;


private:
	std::list<Slot> m_slots_;


public:


	std::vector<result_type> call(std::false_type, Args &&... args) const
	{
		std::vector<result_type> res;

		for (auto const &item:m_slots_)
		{
			if (item)
			{
				res.push_back(item(std::forward<Args>(args) ...));
			}
		}

		return std::move(res);

	}

	void call(std::true_type, Args &&... args) const
	{
		for (auto const &item:m_slots_)
		{
			if (item)
			{
				item(std::forward<Args>(args) ...);
			}
		}
	}

	auto operator()(Args &&... args) const
	DECL_RET_TYPE((call(std::is_same<result_type, void>(), std::forward<Args>(args)...)))



//	result_type collect_result(Args &&...    args) const
//	{
//		auto it = m_slots_.cbegin();
//
//		assert(it != m_slots_.cend())
//
//		result_type res = (*it)(std::forward<Args>(args) ...);
//
//		for (; it != m_slots_.cend(); ++it)
//		{
//			if (*it)
//			{
//				res = (*it)(std::forward<Args>(args) ...);
//			}
//		}
//
//		return res;
//
//	}

	template<typename ...T>
	iterator connect(T &&... args)
	{
		m_slots_.emplace_back(std::forward<T>(args)...);
		return --m_slots_.end();
	}


	void disconnect(iterator it)
	{
		m_slots_.erase(it);
	}


};

template<typename TResult, typename ...Args>
struct Signal<TResult(Args...)>::Slot
{
	struct AbstractTracker
	{
		virtual ~AbstractTracker() { }

		virtual bool expired() const = 0;
	};

	template<typename T>
	struct Tracker : public AbstractTracker
	{

		Tracker(std::shared_ptr<T> &obj) : m_ptr_(obj) { }

		~Tracker() { }

		bool expired() const { return m_ptr_.expired(); }

		std::weak_ptr<T> m_ptr_;
	};

	Slot(Slot &&other) :
			m_tracker_(other.m_tracker_), m_slot_(other.m_slot_)
	{

	}

	Slot(Slot const &other) :
			m_tracker_(other.m_tracker_), m_slot_(other.m_slot_)
	{

	}


	Slot(callback_type const &callback) :
			m_tracker_(nullptr), m_slot_(callback)
	{

	}

	// callable object
	template<typename TFun>
	Slot(TFun const &fun) : m_tracker_(nullptr)
	{
		m_slot_ = [=](Args &&...args)
		{
		    return static_cast<result_type >(fun(std::forward<Args>(args)...));
		};
	}

	// member function
	template<typename T, typename TRes2>
	Slot(std::shared_ptr<T> &o_ptr, TRes2 (T::*pfun)(Args ...))
	{
		m_slot_ = [=](Args &&...args)
		{
		    return static_cast<result_type >((o_ptr.get()->*pfun)(std::forward<Args>(args)...));
		};

		track(o_ptr);
	}

	template<typename TObject>
	void track(std::shared_ptr<TObject> &object)
	{
		m_tracker_ = std::dynamic_pointer_cast<AbstractTracker>(std::make_shared<Tracker<TObject> >(object));
	}

	operator bool() const
	{
		return m_slot_ && (m_tracker_ == nullptr || !m_tracker_->expired());
	}

	result_type operator()(Args &&...args) const
	{
		return m_slot_(std::forward<Args>(args)...);
	}


	std::function<result_type(Args...)> m_slot_;

	std::shared_ptr<AbstractTracker> m_tracker_;

};

}// namespace simpla


>>>>>>> ba2d991a415cad15e2e1816d2f40de9e5eb46fbc
#endif //SIMPLA_SIGNAL_H
