#define INF 1e7
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <iostream>
#include <openacc.h>
//#pragma GCC optimize("O3")
using namespace std;
inline int index(const int i, const int j) {
 return i * SIZE + j;
}
// add your codes begin
float bb[SIZE * SIZE];
int bw[6];
// add your codes end
int main() {
 const int size2 = SIZE * SIZE;
 float* data = new float[size2];
 for (int i = 0; i < size2; i++)
 data[i] = -INF;
 srand(SIZE);
 for (int i = 0; i < SIZE * 20; i++) {
 int prev = rand() % SIZE;
 int next = rand() % SIZE;
 if ((prev == next) || (data[index(prev, next)] > -INF)) {
 i--;
 continue;
 }
 data[index(prev, next)] = log((rand() % 99 + 1.0) / 100);
 }
 copy(data, data + size2, bb);
 double t = omp_get_wtime();
 // add your codes begin 
 
 //int blocksize=SIZE/processor;
 omp_set_num_threads(1);
 int bsize = SIZE / 2;
 bw[0] = 0;
 for (int i = 0; i < 2; i++) {
 bw[i + 1] = bsize * (i + 1);
 }
 #pragma omp parallel
 {
 int my_gpu=omp_get_thread_num();
 acc_set_device_num(my_gpu,acc_device_nvidia);
 // GPU Acceleration with OpenACC
 #pragma acc data copy(bb[0:size2])
 {
 //#pragma omp for schedule(static,1)
 for (int m = 0; m < 2; m++) {
 int is = m * bsize;
 int ie = (m + 1) * bsize;
 
 for (int k = is; k < ie; k++) {
 #pragma acc parallel loop gang worker num_workers(1) vector_length(1024)
 for (int i = is; i < ie; i++) {
 #pragma acc loop vector worker 
 for (int j = is; j < ie; j++) {
    if(i==j) continue;
    if (bb[i * SIZE + j] >= bb[i * SIZE + k] + bb[k * SIZE + j])
        continue;
    bb[i * SIZE + j] = bb[i * SIZE + k] + bb[k * SIZE + j];
 //bb[index(i, j)] = std::max(bb[index(i, j)], bb[index(i, k)] + bb[index(k, j)]);
 }
 }
 }
 
 for (int s1 = 0; s1 < 2; s1++) {
 if (s1 != m) {
 int iss = bw[s1];
 int iee = bw[s1 + 1];
 
 for (int k = is; k < ie; k++) {
 #pragma acc parallel loop gang worker num_workers(1) vector_length(1024)
 for (int i = is; i < ie; i++) {
 #pragma acc loop vector worker 
 for (int j = iss; j < iee; j++) {
    if(i==j) continue;
    if (bb[i * SIZE + j] >= bb[i * SIZE + k] + bb[k * SIZE + j])
        continue;
    bb[i * SIZE + j] = bb[i * SIZE + k] + bb[k * SIZE + j];
 //bb[index(i, j)] = std::max(bb[index(i, j)], bb[index(i, k)] + bb[index(k, j)]);
 }
 }
 }
 }
 }
 
 for (int s1 = 0; s1 < 2; s1++) {
 if (s1 != m) {
 int iss = bw[s1];
 int iee = bw[s1 + 1];
 
 for (int k = is; k < ie; k++) {
 #pragma acc parallel loop gang worker num_workers(1) vector_length(1024)
 for (int i = iss; i < iee; i++) {
 #pragma acc loop vector worker 
 for (int j = is; j < ie; j++) {
    if(i==j) continue;
    if (bb[i * SIZE + j] >= bb[i * SIZE + k] + bb[k * SIZE + j])
        continue;
    bb[i * SIZE + j] = bb[i * SIZE + k] + bb[k * SIZE + j];
 //bb[index(i, j)] = std::max(bb[index(i, j)], bb[index(i, k)] + bb[index(k, j)]);
 }
 }
 }
 }
 }
 
 for (int s1 = 0; s1 < 4; s1++) {
 int x1 = s1 / 2;
 int y1 = s1 - x1 * 2;
 if (x1 != m && y1 != m) {
 int issx = bw[x1];
 int ieex = bw[x1 + 1];
 int issy = bw[y1];
 int ieey = bw[y1 + 1];
 
 for (int k = is; k < ie; k++) {
 #pragma acc parallel loop gang worker num_workers(1) vector_length(1024)
 for (int i = issx; i < ieex; i++) {
 #pragma acc loop vector worker 
 for (int j = issy; j < ieey; j++) {
    if(i==j) continue;
    if (bb[i * SIZE + j] >= bb[i * SIZE + k] + bb[k * SIZE + j])
        continue;
    bb[i * SIZE + j] = bb[i * SIZE + k] + bb[k * SIZE + j];
 //bb[index(i, j)] = std::max(bb[index(i, j)], bb[index(i, k)] + bb[index(k, j)]);
 }
 }
 }
 }
 }
 }
 }
 #pragma acc data copyout(bb[0:size2])
 }
 t = omp_get_wtime() - t;
 printf("time %f %d\n", t, SIZE);
 copy(bb, bb + size2, data);
 for (int i = 0; i < 20; i++) {
 int prev = rand() % SIZE;
 int next = rand() % SIZE;
 if (prev == next) {
 i--;
 continue;
 }
 printf("test %d %d %f\n", prev, next, data[index(prev, next)]);
 }
 delete[] data;
}
