#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdint.h>
#include<stdio.h>

#define gram(__ptr,__ty,__n) cudaMalloc((void**)&__ptr,(__n)*sizeof(__ty))
#define eidx(__i,__j) ((__i)*H+(__j))*2
#define oidx(__i,__j) eidx(__i,__j)+1
#define idx(__i,__j) (__i)*H*2+(__j) //未整除2，不分奇偶性
#include"schro.cuh"
#include"lazy_cheby_exp.h"

//热方程测试通过
__global__ void global_five_point_heat(real* src, real* dst, real a) {
	iint i = blockIdx.x;
	iint j = threadIdx.x;
	bool odd = j % 2;
	j /= 2;
	if (odd) { //虚部
		if (i == 0 || i == W - 1 || j == 0 || j == H - 1) {
			dst[oidx(i, j)] = 0;
		}
		else {
			dst[oidx(i, j)] = (1 - 4 * a) * src[oidx(i, j)] //奇数位--虚部
				+ a * (src[oidx(i, j - 1)] + src[oidx(i, j + 1)]
					+ src[oidx(i - 1, j)] + src[oidx(i + 1, j)]);
		}
	}
	else { //实部
		if (i == 0 || i == W - 1 || j == 0 || j == H - 1) {
			dst[eidx(i, j)] = 0;
		}
		else {
			dst[eidx(i, j)] = (1 - 4 * a) * src[eidx(i, j)] //偶数位--实部
				+ a * (src[eidx(i, j - 1)] + src[eidx(i, j + 1)]
					+ src[eidx(i - 1, j)] + src[eidx(i + 1, j)]);
		}
	}
}

__global__ void global_add(real* dst_a, real* b) {
	iint i = blockIdx.x;
	iint j = threadIdx.x;
	dst_a[idx(i, j)] += b[idx(i, j)];
}
__global__ void global_minus(real* dst_a, real* b) {
	iint i = blockIdx.x;
	iint j = threadIdx.x;
	dst_a[idx(i, j)] -= b[idx(i, j)];
}

#define re eidx(i,j)
#define im oidx(i,j)

__global__ void global_add_i(real* dst_a, real* b) {
	iint i = blockIdx.x;
	iint j = threadIdx.x;
	bool odd = j % 2;
	j /= 2;
	if (odd) {
		dst_a[im] += b[re];
	}
	else {
		dst_a[re] -= b[im];
	}
}
__global__ void global_minus_i(real* dst_a, real* b) {
	iint i = blockIdx.x;
	iint j = threadIdx.x;
	bool odd = j % 2;
	j /= 2;
	if (odd) {
		dst_a[im] -= b[re];
	}
	else {
		dst_a[re] += b[im];
	}
}
__global__ void global_mulnum(real* dst_a, real coef) {
	iint i = blockIdx.x;
	iint j = threadIdx.x;
	dst_a[idx(i, j)] *= coef;
}

inline void gpu_add(real* dst_a, real* b) {
	global_add << <W, 2 * H >> > (dst_a, b);
}
inline void gpu_minus(real* dst_a, real* b) {
	global_minus << <W, 2 * H >> > (dst_a, b);
}
inline void gpu_add_i(real* dst_a, real* b) {
	global_add_i << <W, 2 * H >> > (dst_a, b);
}
inline void gpu_minus_i(real* dst_a, real* b) {
	global_minus_i << <W, 2 * H >> > (dst_a, b);
}
inline void gpu_mulnum(real* dst_a, real coef) {
	global_mulnum << <W, 2 * H >> > (dst_a, coef);
}
//此接口必须禁用
/*
inline void gpu_init(real* obj) {
	gram(obj, real, W * H * 2);
}*/
////////////////
inline void gpu_free(real* obj) {
	cudaFree(obj);
}
inline void gpu_copy(real* src, real* dst) {
	cudaMemcpy(dst, src, DSIZE, cudaMemcpyDeviceToDevice);
}
inline void gpu_output(real* src, real* dst) {
	cudaMemcpy(dst, src, DSIZE, cudaMemcpyDeviceToHost);
}
inline void gpu_upload(real* src, real* dst) {
	cudaMemcpy(dst, src, DSIZE, cudaMemcpyHostToDevice);
}

//五点差分法测试通过
__global__ void global_five_point_legacy(real* src, real* dst, real a) {
	iint i = blockIdx.x;
	iint j = threadIdx.x;
	bool odd = j % 2;
	j /= 2;
	if (odd) { //虚部
		if (i == 0 || i == W - 1 || j == 0 || j == H - 1) {
			dst[oidx(i, j)] = 0;
		}
		else { 
			dst[im] = src[im] - 4 * a * src[re] +
				a * (src[eidx(i - 1, j)] + src[eidx(i + 1, j)] +
					src[eidx(i, j - 1)] + src[eidx(i, j + 1)]);
		}
	}
	else { //实部
		if (i == 0 || i == W - 1 || j == 0 || j == H - 1) {
			dst[eidx(i, j)] = 0;
		}
		else {
			dst[re] = src[re] + 4 * a * src[im] -
				a * (src[oidx(i - 1, j)] + src[oidx(i + 1, j)] +
					src[oidx(i, j - 1)] + src[oidx(i, j + 1)]);
		}
	}
}

__global__ void global_five_point(real* src, real* dst, real a) {
	iint i = blockIdx.x;
	iint j = threadIdx.x;
	bool odd = j % 2;
	j /= 2;
	if (odd) { //虚部
		if (i == 0 || i == W - 1 || j == 0 || j == H - 1) {
			dst[im] = 0;
		}
		else {
			dst[im] = a * (src[oidx(i - 1, j)] + src[oidx(i + 1, j)] +
				src[oidx(i, j - 1)] + src[oidx(i, j + 1)] - 4 * src[im]);
		}
	}
	else { //实部
		if (i == 0 || i == W - 1 || j == 0 || j == H - 1) {
			dst[re] = 0;
		}
		else {
			dst[re] = a * (src[eidx(i - 1, j)] + src[eidx(i + 1, j)] +
				src[eidx(i, j - 1)] + src[eidx(i, j + 1)] - 4 * src[re]);
		}
	}
}

inline void gpu_five_point(real* src, real* dst, real a) {
	global_five_point << <W, 2 * H >> > (src, dst, a);
}
inline void naive_five_point(real* src, real* dst, real a) {
	global_five_point_legacy << <W, 2 * H >> > (src, dst, a);
}

inline void host_naive_five_point(real* src, real* dst, real a) {
	for (int n = 0; n < W * H * 2; n++) {
		iint i = n / (H * 2);
		iint j = n % (H * 2);
		bool odd = j % 2;
		j /= 2;
		if (odd) { //虚部
			if (i == 0 || i == W - 1 || j == 0 || j == H - 1) {
				dst[oidx(i, j)] = 0;
			}
			else {
				dst[im] = src[im] - 4 * a * src[re] +
					a * (src[eidx(i - 1, j)] + src[eidx(i + 1, j)] +
						src[eidx(i, j - 1)] + src[eidx(i, j + 1)]);
			}
		}
		else { //实部
			if (i == 0 || i == W - 1 || j == 0 || j == H - 1) {
				dst[eidx(i, j)] = 0;
			}
			else {
				dst[re] = src[re] + 4 * a * src[im] -
					a * (src[oidx(i - 1, j)] + src[oidx(i + 1, j)] +
						src[oidx(i, j - 1)] + src[oidx(i, j + 1)]);
			}
		}
	}
}

inline void host_five_point(real* src, real* dst, real a) {
	int i, j; bool odd;
	for (int n = 0; n < W * H * 2; n++) {
		i = n / (H * 2);
		j = n % (H * 2);
		odd = j % 2;
		j /= 2;
		if (odd) { //虚部
			if (i == 0 || i == W - 1 || j == 0 || j == H - 1) {
				dst[im] = 0;
			}
			else {
				dst[im] = a * (src[oidx(i - 1, j)] + src[oidx(i + 1, j)] +
					src[oidx(i, j - 1)] + src[oidx(i, j + 1)] - 4 * src[im]);
			}
		}
		else { //实部
			if (i == 0 || i == W - 1 || j == 0 || j == H - 1) {
				dst[re] = 0;
			}
			else {
				dst[re] = a * (src[eidx(i - 1, j)] + src[eidx(i + 1, j)] +
					src[eidx(i, j - 1)] + src[eidx(i, j + 1)] - 4 * src[re]);
			}
		}
	}
}

#define loop(__i,__n) for(int __i=0;__i<__n;__i++)
#define WH2 W*H*2
inline void host_add(real* dst_a, real* b) {
	loop(i, WH2) {
		dst_a[i] += b[i];
	}
}
inline void host_minus(real* dst_a, real* b) {
	loop(i, WH2) {
		dst_a[i] -= b[i];
	}
}
inline void host_add_i(real* dst_a, real* b) {
	bool odd; int i, j;
	loop(n, WH2) {
		i = n / (H * 2);
		j = n % (H * 2);
		odd = j % 2;
		j /= 2;
		if (odd) {
			dst_a[im] += b[re];
		}
		else {
			dst_a[re] -= b[im];
		}
	}
}
inline void host_minus_i(real* dst_a, real* b) {
	bool odd; int i, j;
	loop(n, WH2) {
		i = n / (H * 2);
		j = n % (H * 2);
		odd = j % 2;
		j /= 2;
		if (odd) {
			dst_a[im] -= b[re];
		}
		else {
			dst_a[re] += b[im];
		}
	}
}
inline void host_mulnum(real* dst_a, real coef) {
	loop(i, WH2) {
		dst_a[i] *= coef;
	}
}

__global__ void global_wall(real* dst, int* wall) {
	iint i = blockIdx.x;
	iint j = threadIdx.x;
	if (wall[i * H + j / 2]) {
		dst[idx(i, j)] = 0;
	}
}
inline void gpu_wall(real* dst, int* wall) {
	global_wall << <W, 2 * H >> > (dst, wall);
}
inline void host_wall(real* dst, int* wall) {
	iint i, j;
	loop(n, WH2) {
		i = n / H;
		j = n % H;
		j /= 2;
		if (wall[i * H + j]) {
			dst[eidx(i, j)] = 0;
			dst[eidx(i, j)] = 0;
		}
	}
}