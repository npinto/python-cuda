#ifndef _CUDASVD_KERNEL_H_
#define _CUDASVD_KERNEL_H_

#define ADD     if (tid < 128) { temp[tid] += temp[tid + 128];}	__syncthreads();\
				if (tid <  64) { temp[tid] += temp[tid +  64];}	__syncthreads();\
				if (tid <  32) { temp[tid] += temp[tid +  32];}	__syncthreads();\
				if (tid <  16) { temp[tid] += temp[tid +  16];}	__syncthreads();\
				if (tid <   8) { temp[tid] += temp[tid +   8];}	__syncthreads();\
				if (tid <   4) { temp[tid] += temp[tid +   4];}	__syncthreads();\
				if (tid <   2) { temp[tid] += temp[tid +   2];}	__syncthreads();\
				if (tid <   1) { temp[tid] += temp[tid +   1];}
				 			 				 
__global__ static void bjrot(float * d_w_o, float * d_w_i,float * d_u_o, float * d_u_i, int level, int offset)
{   
	const int tid = threadIdx.x;
	const int tid_second = threadIdx.x + 256;
	const int bstart = (blockIdx.x<<1);
	const int i_index0 = __mul24(bstart + offset,NUM);
	const int i_index1 = __mul24(bstart + 1 + offset,NUM);
	
    __shared__ float value[2][NUM];
	__shared__ float cs;
	__shared__ float ss;
	__shared__ float c;
	__shared__ int o_index0;
	__shared__ int o_index1;
	__shared__ float temp[256];

	if(tid == 0)
	{
		o_index0 = (bstart == ((bstart >> level) << level)) ? bstart : ((bstart + 2) == (((bstart + 2) >> level) << level)) ? (bstart + 1) : (bstart + 2);
		o_index0 = __mul24(o_index0 + offset, NUM);
		o_index1 = (bstart == ((bstart >> level) << level)) ? (bstart + 2) : (bstart - 1);
		o_index1 = __mul24(o_index1 + offset, NUM);
	}
	
	value[0][tid] = d_w_i[i_index0 + tid];
	value[0][tid_second] = d_w_i[i_index0 + tid_second];

	value[1][tid] = d_w_i[i_index1 + tid];
	value[1][tid_second] = d_w_i[i_index1 + tid_second];
					  
	__syncthreads();
	temp[tid] = value[0][tid] * value[0][tid];
	temp[tid] += value[0][tid_second] * value[0][tid_second];
	__syncthreads();
	ADD;
	__syncthreads();
	if(tid == 0)
	cs = temp[0];
	temp[tid] = value[1][tid] * value[1][tid];
	temp[tid] += value[1][tid_second] * value[1][tid_second];
	__syncthreads();
	ADD;
	__syncthreads();
	if(tid == 0)
	ss = temp[0];
	temp[tid] = value[0][tid] * value[1][tid];
	temp[tid] += value[0][tid_second] * value[1][tid_second];
	__syncthreads();
	ADD;
	__syncthreads();
	if(tid == 0)
	c = temp[0];
	__syncthreads();

	if( c > 0.0000001 || c < -0.0000001 )
	{
		if(tid == 0)
		{
			c = ( ss - cs ) * 0.5f / c;
			ss = signbit(c)? -1:1;
			ss /= ( fabsf(c) + sqrtf( 1.0f + c*c));
			cs = rsqrtf(1.0f + ss*ss);
			ss *= cs;
		}
		__syncthreads();

		temp[tid] =  value[0][tid] * cs  - value[1][tid] * ss;
		value[1][tid] *= cs;
		value[1][tid] += value[0][tid] * ss;  
		value[0][tid] = temp[tid];

		temp[tid] =  value[0][tid_second] * cs  - value[1][tid_second] * ss;
		value[1][tid_second] *= cs;
		value[1][tid_second] += value[0][tid_second] * ss; 
		value[0][tid_second] = temp[tid];
	}

	d_w_o[o_index0 + tid] = value[0][tid];
	d_w_o[o_index0 + tid_second] = value[0][tid_second];
	
	d_w_o[o_index1 + tid] = value[1][tid];
	d_w_o[o_index1 + tid_second] = value[1][tid_second];

	if( c > 0.0000001 || c < -0.0000001 )
	{
		value[0][tid] = d_u_i[i_index0 + tid];
		value[0][tid_second] = d_u_i[i_index0 + tid_second];

		value[1][tid] = d_u_i[i_index1 + tid];
		value[1][tid_second] = d_u_i[i_index1 + tid_second];

		temp[tid] =  value[0][tid] * cs - value[1][tid] * ss;
		value[1][tid] *= cs; 
		value[1][tid] += value[0][tid] * ss;
		value[0][tid] = temp[tid];

		temp[tid] =  value[0][tid_second] * cs  - value[1][tid_second] * ss;
		value[1][tid_second] *=  cs;
		value[1][tid_second] +=	value[0][tid_second]  * ss;
		value[0][tid_second] = temp[tid];
	}  		  
	d_u_o[o_index0 + tid] = value[0][tid];
	d_u_o[o_index0 + tid_second] = value[0][tid_second];

	d_u_o[o_index1 + tid] = value[1][tid];
	d_u_o[o_index1 + tid_second] = value[1][tid_second];
}


__global__ static void bjrot8(float * d_w, float * d_u, int loop, int mode)
{   
	const int tid = threadIdx.x;
	const int tid_second = threadIdx.x + 256;
	const int bstart = (blockIdx.x<<1);
	
    __shared__ float value[2][NUM];
	__shared__ float cs;
	__shared__ float ss;
	__shared__ float c;
	__shared__ int itemp;

	__shared__ int index0;
	__shared__ int index1;
	__shared__ int index[7]; 
	__shared__ float temp[256];
	
	index[0] = 1;
	index[1] = 3;
	index[2] = 5;
	index[3] = 7;
	index[4] = 6;
	index[5] = 4;
	index[6] = 2;

	if(tid == 0)
	{
		itemp = bstart%8;
		//printf("%d\n", itemp);
		if (itemp == 0)
		{
			index0 = 0;
			index1 = index[loop];
		}
		else if (itemp == 2)
		{
			index0 = index[(6 + loop)%7];
			index1 = index[(1 + loop)%7];
		}
		else if (itemp == 4)
		{
			index0 = index[(5 + loop)%7];
			index1 = index[(2 + loop)%7];
		}
		else if (itemp == 6)
		{
			index0 = index[(4 + loop)%7];
			index1 = index[(3 + loop)%7];
		}
		itemp = bstart - itemp;
		//printf("%d	%d	",itemp + index0, itemp + index1);
		index0 = __mul24((itemp + index0 + mode * 4), NUM);
		index1 = __mul24((itemp + index1 + mode * 4), NUM);
	}
	__syncthreads();
	value[0][tid] = d_w[index0 + tid];
	value[0][tid_second] = d_w[index0 + tid_second];

	value[1][tid] = d_w[index1 + tid];
	value[1][tid_second] = d_w[index1 + tid_second];
					  
	__syncthreads();
	temp[tid] = value[0][tid] * value[0][tid];
	temp[tid] += value[0][tid_second] * value[0][tid_second];
	__syncthreads();
	ADD;
	__syncthreads();
	if(tid == 0)
	cs = temp[0];
	temp[tid] = value[1][tid] * value[1][tid];
	temp[tid] += value[1][tid_second] * value[1][tid_second];
	__syncthreads();
	ADD;
	__syncthreads();
	if(tid == 0)
	ss = temp[0];
	temp[tid] = value[0][tid] * value[1][tid];
	temp[tid] += value[0][tid_second] * value[1][tid_second];
	__syncthreads();
	ADD;
	__syncthreads();
	if(tid == 0)
	c = temp[0];
	__syncthreads();

	if( c > 0.0000001 || c < -0.0000001 )
	{
		if(tid == 0)
		{
			c = ( ss - cs ) * 0.5f / c;
			ss = signbit(c)? -1:1;
			ss /= ( fabsf(c) + sqrtf( 1.0f + c*c));
			cs = rsqrtf(1.0f + ss*ss);
			ss *= cs;
		}
		__syncthreads();


		temp[tid] =  value[0][tid] * cs  - value[1][tid] * ss;
		value[1][tid] *= cs;
		value[1][tid] += value[0][tid] * ss;  
		value[0][tid] = temp[tid];

		temp[tid] =  value[0][tid_second] * cs  - value[1][tid_second] * ss;
		value[1][tid_second] *= cs;
		value[1][tid_second] += value[0][tid_second] * ss; 
		value[0][tid_second] = temp[tid];

		d_w[index0 + tid] = value[0][tid];
		d_w[index0 + tid_second] = value[0][tid_second];
	
		d_w[index1 + tid] = value[1][tid];
		d_w[index1 + tid_second] = value[1][tid_second];

		value[0][tid] = d_u[index0 + tid];
		value[0][tid_second] = d_u[index0 + tid_second];

		value[1][tid] = d_u[index1 + tid];
		value[1][tid_second] = d_u[index1 + tid_second];

		temp[tid] =  value[0][tid] * cs - value[1][tid] * ss;
		value[1][tid] *= cs; 
		value[1][tid] += value[0][tid] * ss;
		value[0][tid] = temp[tid];

		temp[tid] =  value[0][tid_second] * cs  - value[1][tid_second] * ss;
		value[1][tid_second] *=  cs;
		value[1][tid_second] +=	value[0][tid_second]  * ss;
		value[0][tid_second] = temp[tid];

		d_u[index0 + tid] = value[0][tid];
		d_u[index0 + tid_second] = value[0][tid_second];
	
		d_u[index1 + tid] = value[1][tid];
		d_u[index1 + tid_second] = value[1][tid_second];
	}  
}


#endif // _CUDASVD_KERNEL_H_
