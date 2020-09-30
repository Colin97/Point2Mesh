#include <iostream>
#include "../annoy/src/kissrandom.h"
#include "../annoy/src/annoylib.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <tuple>
#define N 5555555
using namespace std;

std::vector<int> knn_id[N];
std::vector<double> knn_dis[N];
int n_pc, n_candidates = 0;
double average_nn_dis = 0, face_max_l = 1e9;
map<tuple<int, int, int>, bool> face_visit;
AnnoyIndex<int, double, Euclidean, Kiss32Random> pc_knn = AnnoyIndex<int, double, Euclidean, Kiss32Random>(3);

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
}pc[N];

struct Face {
  int a, b, c;
  double d; 
  Face() {};
  Face (int _a, int _b, int _c) {
    a = _a; b = _b; c = _c;
  };
} candidates[N * 10];

bool face_check(int a, int b, int c) {
  int ver[3];
  ver[0] = a; ver[1] = b; ver[2] = c;
  sort(&ver[0], &ver[3]);
  if (face_visit[make_tuple(ver[0], ver[1], ver[2])])
    return false;
  face_visit[make_tuple(ver[0], ver[1], ver[2])] = true;
  return true;
}

int main(int argc, char ** argv) {
  string pc_file = argv[1];
  string candidates_file = argv[2];
  int K = atoi(argv[3]);

  freopen(pc_file.c_str(), "r", stdin);
  scanf("%d", &n_pc);

  double xmin = 1e9, ymin = 1e9, zmin = 1e9;
  double xmax = -1e9, ymax = -1e9, zmax = -1e9;
  for (int i = 0; i < n_pc; i++) {
    double x, y, z;
    scanf("%lf%lf%lf", &x, &y, &z);
    pc[i] = Point(x, y, z);
    xmax = max(xmax, x); ymax = max(ymax, y); zmax = max(zmax, z);
    xmin = min(xmin, x); ymin = min(ymin, y); zmin = min(zmin, z);
  }

  double scale = sqrt((xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin) + (zmax - zmin) * (zmax - zmin));

  for (int i = 0; i < n_pc; i++) {
    pc[i].x = (pc[i].x - (xmax + xmin) / 2) / scale;
    pc[i].y = (pc[i].y - (ymax + ymin) / 2) / scale;
    pc[i].z = (pc[i].z - (zmax + zmin) / 2) / scale;
  }
  
  for (int i = 0; i < n_pc; i++) {
    double vec[3] = {pc[i].x, pc[i].y, pc[i].z};
    pc_knn.add_item(i, vec);   
  }
  pc_knn.build(10);

  for (int i = 0; i < n_pc; i++) {
    pc_knn.get_nns_by_item(i, K + 1, -1, &knn_id[i], &knn_dis[i]);
    average_nn_dis += knn_dis[i][1];
    face_max_l = min(knn_dis[i][K], face_max_l);
  }
  average_nn_dis /= n_pc;
  if (face_max_l < average_nn_dis * 2)
    face_max_l = average_nn_dis * 2;


  for (int i = 0; i < n_pc; i++) {
    for (int j = 1; j <= K; j++)
      for (int k = j + 1; k <= K; k++) {
        int a = i, b = knn_id[i][j], c = knn_id[i][k];
        if (!face_check(a, b, c))
          continue;
        Point A = pc[a];
        Point B = pc[b];
        Point C = pc[c];
        if ((A - B).length() > face_max_l || (A - C).length() > face_max_l || (B - C).length() > face_max_l)
          continue;
        candidates[n_candidates] = Face(a, b, c);
        n_candidates++;
      }
  }

  freopen(candidates_file.c_str(), "w", stdout);
  printf("%d\n", n_pc);
  for (int i = 0; i < n_pc; i++) 
    printf("%lf %lf %lf\n", pc[i].x, pc[i].y, pc[i].z);  
  printf("%d\n", n_candidates);
  for (int i = 0; i < n_candidates; i++) {
    Face f = candidates[i];
    printf("%d %d %d\n", f.a, f.b, f.c);  
  }
  return 0;
}