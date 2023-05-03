#pragma once

#include<cuda_runtime.h>

#define real double	
#define iint int32_t
#define W 256
#define H 256
#define DSIZE (int64_t)W*H*2*sizeof(real)
#define POOL_AMOUNT 16
#define ORDER 20

#define pi 3.14159265358
#define _h (1.0/W)	//ǰ�»��ߣ�С�ĺ���ײ
#define _a 0.1
#define _eta _a*_h*_h
#define _period 2.0/(3.0*_eta*pi) //��������Ե�����
#define _dt  //ʱ�䲽����--����
#define _iter 500 //ÿ�����ִ�м���ʱ�䲽

#define CUDA_AVAIL true

#define safe_free(__ptr) if(__ptr){free(__ptr);__ptr=NULL;}