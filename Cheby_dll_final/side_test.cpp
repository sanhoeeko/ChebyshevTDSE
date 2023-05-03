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

//基本可以运行，若出不演化bug，可用此测试，若能复现，重启电脑可解决问题
void single_pace_five_point(double* data, double* data_g, double* cache_g, double a) {
	gpu_upload(data, data_g);
	for (int t = 0; t < _iter; t++) {
		naive_five_point(data_g, cache_g, a);
		naive_five_point(cache_g, data_g, a);
	}
	gpu_output(data_g, data);
}
//可以运行
void single_pace_five_point_host(double* data, double* cache, double a) {
	for (int t = 0; t < _iter; t++) {
		host_naive_five_point(data, cache, a);
		host_naive_five_point(cache, data, a);
	}
}

void analytic_test_five_point() {
	double* data = cram(double, (int64_t)W * H * 2);
	double* cache = cram(double, (int64_t)W * H * 2);
	double* data_g, * cache_g;
	gram(data_g, double, (int64_t)W * H * 2);
	gram(cache_g, double, (int64_t)W * H * 2);
	data_init(data);
	cout << "周期：" << _period << endl;
	rho_show(data, 0.003, 0);
	double tau = 0;
	for (int t = 0; t < 10; t++) {
		single_pace_five_point(data, data_g, cache_g, _a);
		rho_show(data, 0.003, 0);
		cout << "时间：" << tau << endl;
		cout << cal_rho(data) << endl;
		tau += _dt * _iter;
	}
	free(data); free(cache);
	cudaFree(data_g); cudaFree(cache_g);
}