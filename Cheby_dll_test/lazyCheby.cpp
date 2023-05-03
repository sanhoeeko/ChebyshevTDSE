#include"pch.h"

#include"param.h"
#include"solver.h"
#include"pool.h"
#include"gpu_viewer.h"
#include<iostream>
#include<math.h>
using namespace std;

#define cram(__ty,__n) (__ty*)calloc((__n),sizeof(__ty))
#define eidx(__i,__j) ((__i)*H+(__j))*2
#define oidx(__i,__j) eidx(__i,__j)+1

//全局变量

double tau = 0;
Solver* Sol;
#define sol (*Sol)
double* dat;
double* rho;
double dt = 1;

//////////

inline double f(double x) {
	return sin(pi * x) - 3 * sin(2 * pi * x);
}

void data_init(double* data) {
	double wh = sqrt(W * H);
	for (int i = 1; i < W - 1; i++) { //数据初始化时就留出零边界条件
		for (int j = 1; j < H - 1; j++) {
			double x = (double)i / W;
			double y = (double)j / H;
			data[eidx(i, j)] = f(x) * f(y) / 5 / wh;
			data[oidx(i, j)] = 0;
		}
	}
}
void data_init_test(double* data) {
	for (int i = 0; i < W * H * 2; i += 2) {
		data[i] = 1;
		data[i + 1] = 0;
	}
}

inline double Gauss(double x, double mu, double sig) {
	return exp(-(x - mu) * (x - mu) / (2 * sig * sig)) / pow(2 * pi * sig * sig, 0.25);
}
void data_init_particle(double* data) {
	double p = 0.1;
	double x0 = 0.7;
	double y0 = 0.5;
	double sigma = 0.005;
	double wh = sqrt(W * H);
	for (int i = 1; i < W - 1; i++) { //数据初始化时就留出零边界条件
		for (int j = 1; j < H - 1; j++) {
			double x = (double)i / W;
			double y = (double)j / H;
			data[eidx(i, j)] = Gauss(x, x0, sigma) * Gauss(y, y0, sigma) * cos(p * (x - x0)) / wh;
			data[oidx(i, j)] = Gauss(x, x0, sigma) * Gauss(y, y0, sigma) * sin(p * (x - x0)) / wh;
		}
	}
}

void test_part1() {
	cout << "测试：exp(n)" << endl;
	VPOOL = new Pool<double>(10); //不要忘记初始化pool，否则报this=nullptr错误
	double* data = cram(double, (int64_t)W * H * 2);
	data_init_test(data);
	Sol = new Solver(1, data);
	for (double t = 1; t < 10; t += 1) {
		sol.exp_test1(10, 1);
		sol.get_data();
		double res = sol.u_cpu[0];
		cout << res << " | " << exp(t) << endl;
		if (t != 9)VPOOL->clear_all();
		sol.load_data();
	}
	VPOOL->free_all();
}
void test_part2() {
	cout << "测试：cos(n),sin(n)" << endl;
	VPOOL = new Pool<double>(10);
	double* data = cram(double, (int64_t)W * H * 2);
	data_init_test(data);
	Sol = new Solver(1, data);
	for (double t = 1; t < 10; t += 1) {
		sol.exp_test2(10, 1);
		sol.get_data();
		double res1 = sol.u_cpu[0];
		double res2 = sol.u_cpu[1];
		cout << res1 << ", " << res2 << " | " << cos(t) << ", " << sin(t) << endl;
		if (t != 9)VPOOL->clear_all();
		sol.load_data();
	}
	VPOOL->free_all();
}

void test() {
	test_part1();
	test_part2();
}

void init_pool() {
#if CUDA_AVAIL
	VPOOL = new Pool<double>(POOL_AMOUNT);
#else
	VPOOL = new Pool<double>(POOL_AMOUNT);
#endif
}

void single_pace(Solver& sl, double* data) {
	for (int nn = 0; nn < _iter; nn++) {
		sl.exp(POOL_AMOUNT, dt); //z
		sl.get_data();
		if (nn != _iter - 1)VPOOL->clear_all();
		sl.load_data();
	}
	sl.get_data_to(data);
}

//接口

void end_test() {
	VPOOL->free_all();
}
void set_timeSpan(double t) {
	dt = t / _iter;
}
double getPeriod() {
	return _period;
}

double* analytic_test_init() {
	init_pool();
	cout << "理论周期：" << _period << endl;
	dat = cram(double, (int64_t)W * H * 2);
	rho = cram(double, (int64_t)W * H);
	data_init(dat);
	Sol = new Solver(_a, dat);
	tau = 0;
	return rho; //共享内存
}

double analytic_test_single() {
	single_pace(sol, dat);
	rhos(dat, rho);
	tau += dt * _iter;
	cout << "时间：" << tau << endl;
	return cal_rho(rho);
}