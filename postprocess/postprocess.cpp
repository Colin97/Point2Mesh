#include <iostream>
#include "../annoy/src/kissrandom.h"
#include "../annoy/src/annoylib.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#define N 555555
using namespace std;

std::vector<int> knn_id[N];
std::vector<double> knn_dis[N];
int n_pc, n_candidates;
double average_nn_dis = 0;
map<pair<int, int>, int> edge_cnt;
int last_visit_idx[N * 10];
int visit_idx = 0;
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

    void normalize() {
      double l = length();
      x /= l; y /= l; z /= l;}

    double dot(const Point& v) const {
        return x * v.x + y * v.y + z * v.z;}

    Point cross(const Point& v) const {
        return Point(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x);}
}pc[N];

struct Face {
  int a, b, c;
  int label, id;
  bool in_final_mesh;
  double l;
  Face() {};
  Face (int _a, int _b, int _c) {
    a = _a; b = _b; c = _c;
  };
  bool operator < (const Face& rhs) const {
    if (label == rhs.label)
      return l < rhs.l;
    return label < rhs.label;
  };
} candidates[N * 10];
vector<Face>vertex_faces[N];


bool check_inside(Point A, Point B, Point P, Point Q) {
  // check whether triangle ABQ contain triangle ABP
  Point normal = (Q - A).cross(B - A);
  normal.normalize();
  double d = normal.dot(P - A);
  if (fabs(d) > average_nn_dis * 0.3)
    return false;
  P = P - (normal * d);
  
  double sABP = (P - A).cross(B - A).length();
  double sAQP = (P - A).cross(Q - A).length();
  double sBQP = (Q - B).cross(P - B).length();
  double sABQ = (Q - A).cross(B - A).length();

  if (fabs(sABP + sAQP + sBQP - sABQ) < 1e-5) 
    return true;
  return false;
}

bool seg_seg_intersect(Point A, Point B, Point C, Point D, Point P, Point Q) {
  if (((A - C).length() < 1e-5 && (B - D).length() < 1e-5) ||
      ((A - D).length() < 1e-5 && (B - C).length() < 1e-5)) {
    if (check_inside(A, B, P, Q))
      return true;
    if (check_inside(A, B, Q, P))
      return true;
  }  

  // check whether segment AB and segment CD intersect
  Point norm = (B - A).cross(D - C);
  if (norm.length() < 1e-5) 
    return false;

  norm.normalize();
  double d = norm.dot(C - A);
  if (fabs(d) > average_nn_dis * 0.3)
    return false;
  
  if (fabs(d) > 1e-9) {
    A = A + norm * d;
    B = B + norm * d;
  }
  
  if ((A - C).length() < 1e-5 || (A - D).length() < 1e-5 || (B - C).length() < 1e-5 || (B - D).length() < 1e-5)
    return false;
  
  // area ratio
  Point V1 = (B - A);
  V1.normalize();
  Point V2 = (D - C);
  V2.normalize();
  Point V1V2 = V1.cross(V2);
  Point R1R2 = (C - A);

  double t1, t2;
  if (fabs(V1V2.x) > fabs(V1V2.y) && fabs(V1V2.x) > fabs(V1V2.z)) {
    t1 = (R1R2.cross(V2).x)/(V1V2.x);
    t2 = (R1R2.cross(V1).x)/(V1V2.x);
  }
  else if (fabs(V1V2.y) > fabs(V1V2.x) && fabs(V1V2.y) > fabs(V1V2.z)) {
    t1 = (R1R2.cross(V2).y)/(V1V2.y);
    t2 = (R1R2.cross(V1).y)/(V1V2.y);
  }
  else {
    t1 = (R1R2.cross(V2).z)/(V1V2.z);
    t2 = (R1R2.cross(V1).z)/(V1V2.z);    
  }
  if (t1 < 1e-5 || t1 > (A - B).length() - 1e-5) 
    return false;
  if (t2 < 1e-5 || t2 > (C - D).length() - 1e-5) 
    return false;

  return true;
}

bool tri_tri_intersect(Point A, Point B, Point C, Point P, Point Q, Point R) {
  if (seg_seg_intersect(A, B, P, Q, C, R)) 
    return true;
  if (seg_seg_intersect(A, B, Q, R, C, P)) 
    return true;
  if (seg_seg_intersect(A, B, P, R, C, Q)) 
    return true;
  if (seg_seg_intersect(B, C, P, Q, A, R)) 
    return true;
  if (seg_seg_intersect(B, C, Q, R, A, P)) 
    return true;
  if (seg_seg_intersect(B, C, P, R, A, Q)) 
    return true;
  if (seg_seg_intersect(A, C, P, Q, B, R)) 
    return true;
  if (seg_seg_intersect(A, C, Q, R, B, P)) 
    return true;
  if (seg_seg_intersect(A, C, P, R, B, Q)) 
    return true;
  return false;
}

bool tri_mesh_intersect(int a, int b, int c) {
  visit_idx++;
  int ver[3] = {a, b, c};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 5; j++) 
      for (auto face : vertex_faces[knn_id[ver[i]][j]]) {
        if (last_visit_idx[face.id] == visit_idx)
          continue;
        last_visit_idx[face.id] = visit_idx;
        if (tri_tri_intersect(pc[a], pc[b], pc[c], pc[face.a], pc[face.b], pc[face.c])) 
          return true;
      }
  return false;
}

void edge_add(int a, int b) {
  if (a > b)
    swap(a, b);
  edge_cnt[make_pair(a, b)] += 1;
  return ;
}

bool edge_check(int a, int b) {
  if (a > b)
    swap(a, b);
  return edge_cnt[make_pair(a, b)] < 2;
}

int main(int argc, char ** argv) {
  string input_file = argv[1];
  string output_file = argv[2];

  freopen(input_file.c_str(), "r", stdin);
  scanf("%d", &n_pc);
  for (int i = 0; i < n_pc; i++) {
    double vec[3];
    scanf("%lf%lf%lf", &vec[0], &vec[1], &vec[2]);
    pc[i] = Point(vec[0], vec[1], vec[2]);
    pc_knn.add_item(i, vec);   
  }
  pc_knn.build(10);

  int K = 80;
  for (int i = 0; i < n_pc; i++) {
    pc_knn.get_nns_by_item(i, K + 1, -1, &knn_id[i], &knn_dis[i]);
    average_nn_dis += knn_dis[i][1];

    for (int j = 1; j < K; j++)
      for (int k = j + 1; k < K; k++) {
        Point A = pc[i];
        Point B = pc[knn_id[i][j]];
        Point C = pc[knn_id[i][k]];
        if (fabs((B - C).length() - (A - B).length() - (A - C).length()) < 1e-3) {
          edge_cnt[make_pair(knn_id[i][j], knn_id[i][k])] = 100;
          edge_cnt[make_pair(knn_id[i][k], knn_id[i][j])] = 100;
        }
     }
  }
  average_nn_dis /= n_pc;

  scanf("%d", &n_candidates);
  for (int i = 0; i < n_candidates; i++) {
      int a, b, c, label;
      scanf("%d %d %d %d", &a, &b, &c, &label);
      int ver[3] = {a, b, c};
      sort(&ver[0], &ver[3]);
      a = ver[0]; b = ver[1]; c = ver[2];
      candidates[i] = Face(a, b, c);
      candidates[i].label = label;
      candidates[i].in_final_mesh = false;
      Point A = pc[a];
      Point B = pc[b];
      Point C = pc[c];
      double l1 = (A - B).length();
      double l2 = (A - C).length();
      double l3 = (B - C).length();
      candidates[i].l = max(max(l1, l2), l3);
      candidates[i].id = i;
      Point AC = C - A;
      AC.normalize();
      Point AB = B - A;
      AB.normalize();
      if (AC.cross(AB).length() < 1e-3)
        candidates[i].label = 0;
  }

  sort(&candidates[0], &candidates[n_candidates]);

  int n_faces = 0;
  for (int i = 0; i < n_candidates; i++) {
    Face face = candidates[i];
    
    if (face.label == 0) 
      continue;

    int a = face.a, b = face.b, c = face.c;

    if (!(edge_check(a, b) && edge_check(a, c) && edge_check(b, c))) 
      continue;

    if (tri_mesh_intersect(a, b, c)) 
      continue;
    
    candidates[i].in_final_mesh = true;
    n_faces++;
    vertex_faces[a].push_back(face);
    vertex_faces[b].push_back(face);
    vertex_faces[c].push_back(face);
    edge_add(a, b); edge_add(a, c); edge_add(b, c);
  }

  freopen(output_file.c_str(), "w", stdout);
  printf("ply\n");
  printf("format ascii 1.0\n");
  printf("element vertex %d\n", n_pc);
  printf("property float x\n");
  printf("property float y\n");
  printf("property float z\n");
  printf("element face %d\n", n_faces);
  printf("property list uchar int vertex_indices\n");
  printf("end_header\n");

  for (int i = 0; i < n_pc; i++)
    printf("%lf %lf %lf\n", pc[i].x, pc[i].y, pc[i].z);
  
  for (int i = 0; i < n_candidates; i++)
    if (candidates[i].in_final_mesh)
      printf("3 %d %d %d\n", candidates[i].a, candidates[i].b, candidates[i].c);
  return 0;
}