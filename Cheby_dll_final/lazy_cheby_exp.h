#pragma once

#include"bessel.h"

template<typename state_type, typename aux_type> //已测试
class Cheby {
public:
	state_type x0, x1, x2;
	aux_type parameter;
	Cheby(state_type x, state_type e, aux_type para) {
		x1 = x; x0 = e;
		parameter = para;
	}
	~Cheby() {
		;
	}
	/* //////////////////////////////
		数据结构说明
	(1)------------- after: x2=temp;
		x0 -> ■
		x1 -> ■
		x2 -> ■ <- temp
	(2)------------- after: x0.die();x0=x1;x1=x2;
		      □
		x0 -> ■
	 x1,x2 -> ■
	(3)------------- after: x2=temp;
			  □
		x0 -> ■
		x1 -> ■
		x2 -> ■ <- temp
	只有这样安排才能既不泄漏内存，又不误删数据。
	////////////////////////////// */
	state_type iter(state_type func(state_type&, aux_type&)) {
		//函数指针：func(y,x)相当于x*y
		auto temp = func(x1, parameter); //2在这里乘，虽然牺牲了性能，但可以有效防止设计func时的失误
		temp.mulnum(2); 
		x2 = temp;
		x2 -= x0;
		x0.die();
		x0 = x1;
		x1 = x2;
		return x2.copy(); //这里若不隔离，后面使用mulnum时会污染
	}
	void die() { //一旦使用了内存池，各级对象都要具有die()方法
		x0.die();
		x1.die();
		x2.die();
	}
};


template<typename ty, typename aux_ty>
ty chebyExp(ty func(ty&, aux_ty&), Cheby<ty, aux_ty>& che, double z, int order) {
	auto bsi = BesselI_array(z, order * 4);
	ty res = che.x0 * (bsi.val[0] / 2);
	auto x1 = che.x1 * bsi.val[1]; //不能直接使用"+="，否则虚拟内存泄漏
	res += x1;
	x1.die();
	for (int i = 2; i < order; i++) {
		auto temp = che.iter(func);
		temp.mulnum(bsi.val[i]); //不能直接乘，否则泄漏
		res += temp;
		temp.die();
	}
	return res * 2;
}

template<typename ty, typename aux_ty>
ty chebyImagExp(ty func(ty&, aux_ty&), Cheby<ty, aux_ty>& che, double z, int order) {
	auto bsj = BesselJ_array(z, order * 4);
	ty res = che.x0 * (bsj.val[0] / 2);
	auto x1 = che.x1 * bsj.val[1];
	x1.mul_nega_i(1);
	res += x1;
	x1.die();
	for (int i = 2; i < order; i++) {
		auto xi = che.iter(func);
		xi.mul_nega_i(i);
		auto temp =  xi * bsj.val[i];
		res += temp;
		temp.die();
		xi.die();
	}
	return res * 2;
}