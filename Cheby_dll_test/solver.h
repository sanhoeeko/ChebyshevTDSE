#pragma once

#include<stdio.h>
#include<stdlib.h>
#include"param.h"
#include"schro.cuh"
#include"lazy_cheby_exp.h"
#include"pool.h"
#include<vector>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdint.h>
#include"gpu_viewer.h"

#define cram(__ty,__n) (__ty*)calloc((__n),sizeof(__ty))
#define gram(__ptr,__ty,__n) cudaMalloc((void**)&__ptr,(__n)*sizeof(__ty))

inline char mod(char x, char y) {
	if (x % y) {
		if (x < 0) {
			return y + x % y;
		}
		return x % y;
	}
	return 0;
}

class GpuMat {
public:
	real* gpu_ptr;
	GpuMat(real* loc) {
		gpu_ptr = loc;
		if (gpu_ptr == NULL) { //�����п��ܳ������Ų�
			printf("cudaMalloc failed!\n");
		}
	}
	~GpuMat() {
		; //��д�����ͷ�
	}
	real* operator*() {
		return gpu_ptr;
	}
};

enum Phase:char {
	ONE, AI, NEGA, NAI
};

class CplxField { //ż��λʵ��������λ�鲿
public:
	Pool<real>* pool;
	iptr<real> ptr;
	GpuMat* field; //�൱��real**
	Phase phase;
	CplxField(){}
	CplxField(Pool<real>* Pool) { //�����Լ������ڴ棬ֻ������
		pool = Pool;
		ptr = Pool->apply();
		field = new GpuMat(ptr.loc);
		phase = ONE;
	}
	CplxField(Pool<real>* Pool, real* data) { //ͬʱ��gpu�˸������ݵĹ��캯��
		pool = Pool;
		ptr = Pool->apply();
		field = new GpuMat(ptr.loc);
#if CUDA_AVAIL
		gpu_upload(data, **field);
#else
		memcpy(**field, data, DSIZE);
#endif
		phase = ONE;
	}
	~CplxField() {
		;
	}
	void die() { //���������ڴ�
		pool->del(ptr);
	}
	void turnTo(const CplxField& a) {
		pool->del(ptr);
		pool = a.pool;
		ptr = a.ptr;
		field = a.field;
		phase = a.phase;
	}
	void show() {
#if CUDA_AVAIL
		g_show(**field);
#else
		c_show(**field);
#endif
	}
	CplxField(const CplxField& a) { //�Զ�������ǳ������������Դ��
		pool = a.pool;
		ptr = a.ptr;
		field = a.field;
		phase = a.phase;
	}
	CplxField copy() { //���
		CplxField res = CplxField(pool);
#if CUDA_AVAIL
		gpu_copy(**field, **res.field);
#else
		memcpy(**res.field, **field, DSIZE);
#endif
		res.phase = phase;
		return res;
	}
	void mulnum(real a) {
#if CUDA_AVAIL
		gpu_mulnum(**field, a);
#else
		host_mulnum(**field, a);
#endif
	}
	CplxField operator*(real a) {
		CplxField res = copy();
		res.mulnum(a);
		return res;
	}
	void operator+=(const CplxField& a) {
		char diff_phase = mod((phase - a.phase), 4);
		switch (diff_phase)
		{
#if CUDA_AVAIL
		case 0:gpu_add(**field, **a.field); return;
		case 1:gpu_add_i(**field, **a.field); return;
		case 2:gpu_minus(**field, **a.field); return;
		case 3:gpu_minus_i(**field, **a.field); return;
#else
		case 0:host_add(**field, **a.field); return;
		case 1:host_add_i(**field, **a.field); return;
		case 2:host_minus(**field, **a.field); return;
		case 3:host_minus_i(**field, **a.field); return;
#endif
		default:cout <<"Phase error!" << endl; return;
		}
	}
	void operator-=(const CplxField& a) {
		char diff_phase = mod((phase - a.phase), 4);
		switch (diff_phase)
		{
#if CUDA_AVAIL
		case 0:gpu_minus(**field, **a.field); return;
		case 1:gpu_minus_i(**field, **a.field); return;
		case 2:gpu_add(**field, **a.field); return;
		case 3:gpu_add_i(**field, **a.field); return;
#else
		case 0:host_minus(**field, **a.field); return;
		case 1:host_minus_i(**field, **a.field); return;
		case 2:host_add(**field, **a.field); return;
		case 3:host_add_i(**field, **a.field); return;
#endif
		default:cout << "Phase error!" << endl; return;
		}
	}
	void forward(CplxField& dst, real a) {
#if CUDA_AVAIL
		gpu_five_point(**field, **dst.field, a);
#else
		host_five_point(**field, **dst.field, a);
#endif
	}
	void mul_nega_i(int power) {
		phase = Phase(mod((phase - power), 4));
	}
};

inline CplxField forward_new(CplxField& x, real& a) {
	CplxField res(x.pool); //����Ĺ������������ڴ�أ�����new
	x.forward(res, a);
	return res;
}

inline CplxField exp_test(CplxField& x, real& a) {
	real x0 = 1; //�������x
	return x * (a * x0);
}

Pool<real>* v_pool;
#define VPOOL v_pool

class Solver {
public:
	CplxField* u;
	real a;
	real* u_cpu;
	Solver(){}
	Solver(real a, real* data) {
		this->a = a;
		u = new CplxField(v_pool, data); //��Ҫ����new������������˸���įbug��
		u_cpu = cram(real, (int64_t)W * H * 2); //Solver����Ҫ������Ǩ���ʲ����ڴ�ع���
	}
	Solver(real a) {
		this->a = a;
		u = new CplxField(v_pool);
		u_cpu = cram(real, (int64_t)W * H * 2);
	}
	~Solver() {
		free(u_cpu);
		delete u;
	}
	/*�����б�
		a  ����ַ��е���ɢϵ��
		z  ʱ���йص�ϵ��
	*/
	void exp(int order, real z) {
		auto temp_e = u->copy();
		auto temp_x = forward_new(*u, a);
		Cheby<CplxField, real> che(temp_x, temp_e, a);
		u->turnTo(chebyImagExp<CplxField>(forward_new, che, z, order));
		che.die();
		temp_x.die();
		temp_e.die();
	}
	void get_data() {
#if CUDA_AVAIL
		gpu_output(**u->field, u_cpu);
#else
		memcpy(u_cpu, **u->field, DSIZE);
#endif
	}
	void get_data_to(real* data) {
#if CUDA_AVAIL
		gpu_output(**u->field, data);
#else
		memcpy(data, **u->field, DSIZE);
#endif
	}
	void load_data() {
		u = new CplxField(v_pool, u_cpu);
	}
	void load_data_from(real* data) {
		u = new CplxField(v_pool, data);
	}
	void say_hello() {
		printf("Solver:\n");
		printf("gpu���ݴ洢�������ַ��%d\n", u->ptr.idx);
		printf("gpu���ݴ洢��ʵ�ʵ�ַ��%x\n", u->ptr.loc);
		printf("cpu���ݴ洢�ڣ�%x\n", u_cpu);
	}
	void exp_test1(int order, real z) {
		// ����1 -- ����exp(n)
		auto temp_x = *u * a;
		auto temp_e = u->copy(); //������Ҫ������u
		Cheby<CplxField, real> che(temp_x, temp_e, a);
		u->turnTo(chebyExp<CplxField>(exp_test, che, z, order));
		che.die();
		temp_x.die();
		temp_e.die();
	}
	void exp_test2(int order, real z) {
		// ����2 -- ����cos(n),sin(n)
		auto temp_x = *u * a;
		auto temp_e = u->copy();
		Cheby<CplxField, real> che(temp_x, temp_e, a);
		u->turnTo(chebyImagExp<CplxField>(exp_test, che, z, order));
		che.die();
		temp_x.die();
		temp_e.die();
	}
};

real cal_rho(real* rho) {
	real s = 0;
	for (int n = 0; n < (int64_t)W * H; n++) {
		s += rho[n];
	}
	return s;
}

void rhos(real* mat, real* rho) {
	real a, b;
	for (int i = 0; i < W; i++) {
		for (int j = 0; j < H; j++) {
			a = mat[eidx(i, j)];
			b = mat[oidx(i, j)];
			rho[i * H + j] = a * a + b * b;
		}
	}
}