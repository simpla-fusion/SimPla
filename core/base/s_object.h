//
// Created by salmon on 7/10/15.
//

#ifndef SIMPLA_S_OBJECT_H
#define SIMPLA_S_OBJECT_H
namespace simpla
{

struct SObject
{

};


} // namespace simpla

#include <QtCore>

class Goo : public QObject
{
	Q_OBJECT
public:
	Goo()
	{
		connect(this, &Goo::someSignal, this, &Goo::someSlot1); //error
		connect(this, &Goo::someSignal, this, &Goo::someSlot2); //works
	}

	signals:
	void
	someSignal(QString);
public:
	void someSlot1(int);

	void someSlot2(QVariant);
};

#endif //SIMPLA_S_OBJECT_H
