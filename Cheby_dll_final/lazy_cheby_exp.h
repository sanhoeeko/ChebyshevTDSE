#pragma once

#include"bessel.h"

template<typename state_type, typename aux_type> //�Ѳ���
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
		���ݽṹ˵��
	(1)------------- after: x2=temp;
		x0 -> ��
		x1 -> ��
		x2 -> �� <- temp
	(2)------------- after: x0.die();x0=x1;x1=x2;
		      ��
		x0 -> ��
	 x1,x2 -> ��
	(3)------------- after: x2=temp;
			  ��
		x0 -> ��
		x1 -> ��
		x2 -> �� <- temp
	ֻ���������Ų��ܼȲ�й©�ڴ棬�ֲ���ɾ���ݡ�
	////////////////////////////// */
	state_type iter(state_type func(state_type&, aux_type&)) {
		//����ָ�룺func(y,x)�൱��x*y
		auto temp = func(x1, parameter); //2������ˣ���Ȼ���������ܣ���������Ч��ֹ���funcʱ��ʧ��
		temp.mulnum(2); 
		x2 = temp;
		x2 -= x0;
		x0.die();
		x0 = x1;
		x1 = x2;
		return x2.copy(); //�����������룬����ʹ��mulnumʱ����Ⱦ
	}
	void die() { //һ��ʹ�����ڴ�أ���������Ҫ����die()����
		x0.die();
		x1.die();
		x2.die();
	}
};


template<typename ty, typename aux_ty>
ty chebyExp(ty func(ty&, aux_ty&), Cheby<ty, aux_ty>& che, double z, int order) {
	auto bsi = BesselI_array(z, order * 4);
	ty res = che.x0 * (bsi.val[0] / 2);
	auto x1 = che.x1 * bsi.val[1]; //����ֱ��ʹ��"+="�����������ڴ�й©
	res += x1;
	x1.die();
	for (int i = 2; i < order; i++) {
		auto temp = che.iter(func);
		temp.mulnum(bsi.val[i]); //����ֱ�ӳˣ�����й©
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