#include <stdio.h>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>

#define N 1111111
#define B 1
#define MESH 5555555

__global__ void NmDistanceKernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){
	const int batch=512;
	__shared__ float buf[batch*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=batch){
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*3+j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				float x1=xyz[(i*n+j)*3+0], y1=xyz[(i*n+j)*3+1], z1=xyz[(i*n+j)*3+2];
				int best_i=0;
				float best=0;
				int end_ka=end_k-(end_k&3);
				for (int k=0;k<end_ka;k+=4){
					{
						float x2=buf[k*3+0]-x1, y2=buf[k*3+1]-y1, z2=buf[k*3+2]-z1;
						float d=x2*x2+y2*y2+z2*z2;
						if (k==0 || d<best){
							best=d; best_i=k+k2;}
					}
					{
						float x2=buf[k*3+3]-x1, y2=buf[k*3+4]-y1, z2=buf[k*3+5]-z1;
						float d=x2*x2+y2*y2+z2*z2;
						if (d<best){
							best=d; best_i=k+k2+1;}
					}
					{
						float x2=buf[k*3+6]-x1, y2=buf[k*3+7]-y1, z2=buf[k*3+8]-z1;
						float d=x2*x2+y2*y2+z2*z2;
						if (d<best){
							best=d; best_i=k+k2+2;}
					}
					{
						float x2=buf[k*3+9]-x1, y2=buf[k*3+10]-y1, z2=buf[k*3+11]-z1;
						float d=x2*x2+y2*y2+z2*z2;
						if (d<best){
							best=d; best_i=k+k2+3;}
					}
				}
				for (int k=end_ka;k<end_k;k++){
					float x2=buf[k*3+0]-x1, y2=buf[k*3+1]-y1, z2=buf[k*3+2]-z1;
					float d=x2*x2+y2*y2+z2*z2;
					if (k==0 || d<best){
						best=d; best_i=k+k2;}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}
void chamfer_cuda_forward(int b, int n, float * xyz1, int m, float * xyz2, float * dist1, int * idx1,float * dist2, int * idx2, cudaStream_t stream){
    NmDistanceKernel<<<dim3(32,16,1),512>>>(b, n, xyz1, m, xyz2, dist1, idx1);
    cudaDeviceSynchronize();
    NmDistanceKernel<<<dim3(32,16,1),512>>>(b, m, xyz2, n, xyz1, dist2, idx2);
    cudaDeviceSynchronize();
	return ;
}


float xyz1[B][N][3], xyz2[B][N][3];
float dist1[B][N], dist2[B][N];
int idx1[B][N], idx2[B][N];

float *xyz1_gpu, *xyz2_gpu, *dist1_gpu, *dist2_gpu;
int *idx1_gpu, *idx2_gpu;


struct Point {
    double x, y, z;
    Point() {};
    Point (double _x, double _y, double _z) {
      x = _x; y = _y; z = _z;
    };
    Point operator - (const Point& v) const {
          return Point(x - v.x, y - v.y, z - v.z);}
  
      Point operator + (const Point& v) const {
          return Point(x + v.x, y + v.y, z + v.z);}
  
      Point operator * (const double t) const {
        return Point(x * t, y * t, z * t);}
  
      double length() {
        return sqrt(x * x + y * y + z * z);}
  
      void normalize() {
        double l = length();
        x /= l; y /= l; z /= l;}
  
      float dot(const Point& v) const {
          return x * v.x + y * v.y + z * v.z;}
  
      Point cross(const Point& v) const {
          return Point(
              y * v.z - z * v.y,
              z * v.x - x * v.z,
              x * v.y - y * v.x);}
  
}vertices1[MESH], vertices2[MESH], normal1[MESH], normal2[MESH];

struct Face {
    int a, b, c;
    double s; 
    Face() {};
    Face (int _a, int _b, int _c) {
      a = _a; b = _b; c = _c;
    };
}faces1[MESH], faces2[MESH];

int n_vertices_1, n_vertices_2, n_faces_1, n_faces_2;
int n = 0, m = 0;
int resolution = 1000000;

Point randomPointTriangle(Point a, Point b, Point c) {
    double r1 = (double) rand() / RAND_MAX;
    double r2 = (double) rand() / RAND_MAX;
    double r1sqr = std::sqrt(r1);
    double OneMinR1Sqr = (1 - r1sqr);
    double OneMinR2 = (1 - r2);
    a = a * OneMinR1Sqr;
    b = b * OneMinR2;
    return  (c * r2 + b) * r1sqr + a;
}

int main(int argc, char ** argv) {
    std::string mesh1_file = argv[1];
    std::string mesh2_file = argv[2];
    std::string model_id = argv[3];

    freopen(mesh1_file.c_str(), "r", stdin);
    scanf("%d%d", &n_vertices_1, &n_faces_1);
    for (int i = 0; i < n_vertices_1; i++) {
        double x, y, z;
        scanf("%lf %lf %lf", &x, &y, &z);
        vertices1[i] = Point(x, y, z);
    }
    double sum_area = 0;
    for (int i = 0; i < n_faces_1; i++) {
        int _, a, b, c;
        scanf("%d %d %d %d", &_, &a, &b, &c);
        faces1[i] = Face(a, b, c);
        faces1[i].s = (vertices1[c] - vertices1[a]).cross((vertices1[b] - vertices1[a])).length() / 2;
	    if (std::isnan(faces1[i].s)) 
            faces1[i].s=0;
        sum_area += faces1[i].s;
    }
    for (int i = 0; i < n_faces_1; i++) {
        int a = faces1[i].a, b = faces1[i].b, c = faces1[i].c;
        int t = round(resolution * (faces1[i].s / sum_area)); 
        Point normal = (vertices1[c] - vertices1[a]).cross(vertices1[b] - vertices1[a]);
        normal.normalize();
        for (int j = 0; j < t; j++) {
            Point p = randomPointTriangle(vertices1[a], vertices1[b], vertices1[c]);
            xyz1[0][n][0] = p.x; xyz1[0][n][1] = p.y; xyz1[0][n][2] = p.z;
            normal1[n] = normal;
            n++;
        }
    }

    freopen(mesh2_file.c_str(), "r", stdin);
    scanf("%d%d", &n_vertices_2, &n_faces_2);
    for (int i = 0; i < n_vertices_2; i++) {
        double x, y, z;
        scanf("%lf %lf %lf", &x, &y, &z);
        vertices2[i] = Point(x, y, z);
    }
    sum_area = 0;
    for (int i = 0; i < n_faces_2; i++) {
        int _, a, b, c;
        scanf("%d %d %d %d", &_, &a, &b, &c);
        faces2[i] = Face(a, b, c);
        faces2[i].s = (vertices2[c] - vertices2[a]).cross((vertices2[b] - vertices2[a])).length() / 2;
        sum_area += faces2[i].s;
    }
    for (int i = 0; i < n_faces_2; i++) {
        int a = faces2[i].a, b = faces2[i].b, c = faces2[i].c;
        int t = round(resolution * (faces2[i].s / sum_area));
        Point normal = (vertices2[c] - vertices2[a]).cross(vertices2[b] - vertices2[a]);
        normal.normalize(); 
        for (int j = 0; j < t; j++) {
            Point p = randomPointTriangle(vertices2[a], vertices2[b], vertices2[c]);
            xyz2[0][m][0] = p.x; xyz2[0][m][1] = p.y; xyz2[0][m][2] = p.z;
            normal2[m] = normal;
            m++;
        }
    }
    
    size_t xyz_size = max(n, m) * 3 * sizeof(float);
    size_t dis_size = max(n, m) * sizeof(float);
    size_t idx_size = max(n, m) * sizeof(int);
    cudaMalloc((void **) &xyz1_gpu, xyz_size);
    cudaMalloc((void **) &xyz2_gpu, xyz_size);
    cudaMalloc((void **) &dist1_gpu, dis_size);
    cudaMalloc((void **) &dist2_gpu, dis_size);
    cudaMalloc((void **) &idx1_gpu, idx_size);
    cudaMalloc((void **) &idx2_gpu, idx_size);

    cudaMemcpy(xyz1_gpu, &xyz1[0][0], xyz_size, cudaMemcpyHostToDevice);
    cudaMemcpy(xyz2_gpu, &xyz2[0][0], xyz_size, cudaMemcpyHostToDevice);

    chamfer_cuda_forward(1, n, xyz1_gpu, m, xyz2_gpu, dist1_gpu, idx1_gpu, dist2_gpu, idx2_gpu, NULL);

    cudaMemcpy(&dist1[0][0], dist1_gpu, dis_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&dist2[0][0], dist2_gpu, dis_size, cudaMemcpyDeviceToHost);

    cudaMemcpy(&idx1[0][0], idx1_gpu, idx_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&idx2[0][0], idx2_gpu, idx_size, cudaMemcpyDeviceToHost);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
	    printf("error in nnd updateOutput: %s\n", cudaGetErrorString(err));
	    return 0;
	}

    double sum = 0;

    double sum_normal = 0;

    // normal consistency
    for (int i = 0; i < n; i++) {
        sum_normal += abs(normal1[i].dot(normal2[idx1[0][i]]));
    }

    for (int i = 0; i < m; i++) {
         sum_normal += abs(normal2[i].dot(normal1[idx2[0][i]]));
    }

    // f-score for different threshold
    for (int k = 0; k <= 40; k++) {
	double threashold = sqrt(sum_area / resolution) * (1.0 + (double)k / 20);
    	int cnt1 = n, cnt2 = m;
    	for (int i = 0; i < n; i++) {
        	double d = sqrt(dist1[0][i]);
        	if (d > threashold)
            		cnt1--;
        	if (k == 0) sum += d;        
    	}
    	for (int i = 0; i < m; i++) {
        	double d = sqrt(dist2[0][i]);
        	if (d > threashold)
            		cnt2--;
        	if (k == 0) sum += d;        
    	}
    	double t1 = (double) cnt1 / n;
    	double t2 = (double) cnt2 / m;
    	double f1 = 2 * t1 * t2 / (t1 + t2 + 1e-9);
        printf("%lf ", f1);
    }

    // chamfer distance & normal consistency
    printf("%lf %lf %s\n", sum / (n + m), sum_normal / (n + m), model_id.c_str());
    return 0;
}
