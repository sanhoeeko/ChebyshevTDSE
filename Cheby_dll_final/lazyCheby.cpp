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
int* wall; //实际上当做bool* 来用
int* wall_g;
double dt = 1;
int iter = 100;
double momentum = H * pi;
double sig = 0.01;

//////////
void data_init_test(double* data) {
	for (int i = 0; i < W*H*2; i+=2) {
		data[i] = 1;
		data[i + 1] = 0;
	}
}

inline double Gauss(double x, double mu, double sig) {
	return exp(-(x - mu) * (x - mu) / (2 * sig * sig)) / pow(4 * pi * sig * sig, 0.25);
}
void data_init_particle(double* data) {
	double p = momentum;
	double x0 = 0.3;
	double y0 = 0.25;
	double sigma = sig;
	double wh = sqrt(W * H);
	for (int i = 1; i < W - 1; i++) { //数据初始化时就留出零边界条件
		for (int j = 1; j < H - 1; j++) {
			double x = (double)i / W;
			double y = (double)j / H * 0.5;
			data[eidx(i, j)] = Gauss(x, x0, sigma) * Gauss(y, y0, sigma) * cos(p * (x-x0)) / wh;
			data[oidx(i, j)] = Gauss(x, x0, sigma) * Gauss(y, y0, sigma) * sin(p * (x-x0)) / wh;
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

// 接口
#include"csvio.h"
#include<cuda_runtime.h>

#if CUDA_AVAIL
#define wal wall_g
#else
#define wal wall
#endif
void single_pace_main_proj(Solver& sl, double* data) {
	for (int nn = 0; nn < iter; nn++) {
		sl.exp(ORDER, dt); //z
		sl.wall(wal);
		sl.get_data();
		if (nn != _iter - 1)VPOOL->clear_all();
		sl.load_data();
	}
	sl.get_data_to(data);
}

void init_wall() {
	wall = cram(int, W * H);
	gram(wall_g, int, W * H);
	csv_read<int>("wall.csv", wall, W * H);
	cudaMemcpy(wall_g, wall, W * H * sizeof(int), cudaMemcpyHostToDevice);
}

double* main_proj_init() {
	init_pool();
	init_wall();
	dat = cram(double, (int64_t)W * H * 2);
	rho = cram(double, (int64_t)W * H);
	data_init_particle(dat);
	Sol = new Solver(_a, dat);
	tau = 0;
	return rho;
}

double main_proj_single() {
	single_pace_main_proj(sol, dat);
	//rho_show(dat, 2e-3, 0, wall);
	rhos(dat, rho);
	tau += dt * iter;
	cout << "时间：" << tau << endl;
	return cal_rho(rho);
}

void end_test() {
	VPOOL->free_all();
}

void set_timeSpan(double t) {
	dt = t / iter;
}
void set_iter(int n) {
	int old_iter = iter;
	iter = n;
	dt = dt * old_iter / iter; //勿忘更新依赖变量！
}
void set_momentum(double p) {
	momentum = p;
}
void set_sigma(double s) {
	sig = s;
}
void seeParams() {
	cout << "p = " << momentum << endl;
	cout << "sigma = " << sig << endl;
	cout << "dt*iter = " << dt * iter << endl;
}