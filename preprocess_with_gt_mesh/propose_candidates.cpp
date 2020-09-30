#include <iostream>
#include "../annoy/src/kissrandom.h"
#include "../annoy/src/annoylib.h"
#include "octree.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <tuple>
#define N 5555555
using namespace std;

std::vector<int> knn_id[N];
std::vector<double> knn_dis[N];
int n_vertices, n_faces, n_pc, n_candidates = 0;
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
}vertices[N], pc[N];

struct Face {
  int a, b, c;
  double d; 
  Face() {};
  Face (int _a, int _b, int _c) {
    a = _a; b = _b; c = _c;
  };
}faces[N], candidates[N * 10];

bool face_check(int a, int b, int c) {
  int ver[3];
  ver[0] = a; ver[1] = b; ver[2] = c;
  sort(&ver[0], &ver[3]);
  if (face_visit[make_tuple(ver[0], ver[1], ver[2])])
    return false;
  face_visit[make_tuple(ver[0], ver[1], ver[2])] = true;
  return true;
}

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

bool SameSide(Point A, Point B, Point C, Point P) {
    Point v1 = (B - A).cross(C - A);
    Point v2 = (B - A).cross(P - A);
    return v1.dot(v2) >= 0;
}

bool PointinTriangle(Point A, Point B, Point C, Point P) {
    return SameSide(A, B, C, P) && SameSide(B, C, A, P) && SameSide(C, A, B, P);
}

double disPointSegment(Point P, Point A, Point B) {
  double lAB = (A - B).length();
  double r = (P - A).dot(B - A) / (lAB * lAB);
  if (r < 0) 
    return (A - P).length();
  else if (r > 1)
    return (B - P).length();
  else
    return (A + ((B - A) * r) - P).length();
}

double disPointTriangle(Point P, Point A, Point B, Point C) {
  Point normal = (B - A).cross(C - A);
  normal.normalize();
  double t = (A - P).dot(normal);
  Point Q = P + (normal * t);
  if (PointinTriangle(A, B, C, Q)) 
    return (Q - P).length();
  double dAB = disPointSegment(P, A, B);
  double dAC = disPointSegment(P, A, C);
  double dBC = disPointSegment(P, B, C);
  return min(min(dAB, dAC), dBC);
}

double PointMeshDis(Point p, double r, Octree *root) {
  double query_l[3], query_r[3], min_dis = 1e9;
  query_l[0] = p.x - r; query_l[1] = p.y - r; query_l[2] = p.z - r;
  query_r[0] = p.x + r; query_r[1] = p.y + r; query_r[2] = p.z + r;
  set<int> face_id;
  root->query(query_l, query_r, &face_id);
  for (int id: face_id) {
    Face face = faces[id];
    min_dis = min(min_dis, disPointTriangle(p, vertices[face.a], vertices[face.b], vertices[face.c]));
  }
  return min_dis;
}

int main(int argc, char ** argv) {
  string pc_file = argv[1];
  string mesh_file = argv[2];
  string candidates_file = argv[3];
  int K = atoi(argv[4]);

  freopen(pc_file.c_str(), "r", stdin);
  scanf("%d", &n_pc);
  for (int i = 0; i < n_pc; i++) {
    double vec[3];
    scanf("%lf%lf%lf", &vec[0], &vec[1], &vec[2]);
    pc[i] = Point(vec[0], vec[1], vec[2]);
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

  freopen(mesh_file.c_str(), "r", stdin);
  scanf("%d%d", &n_vertices, &n_faces);
  double coor_min = 1e9, coor_max = -1e9;
  for (int i = 0; i < n_vertices; i++) {
    double x, y, z;
    scanf("%lf%lf%lf", &x, &y, &z);
    vertices[i] = Point(x, y, z);
    coor_max = max(coor_max, x); coor_max = max(coor_max, y); coor_max = max(coor_max, z);
    coor_min = min(coor_min, x); coor_min = min(coor_min, y); coor_min = min(coor_min, z);
  }

  double range_l[3] = {coor_min - 0.1, coor_min - 0.1, coor_min - 0.1};
  double range_r[3] = {coor_max + 0.1, coor_max + 0.1, coor_max + 0.1};
  Octree *root = new Octree(range_l, range_r, average_nn_dis * 2);

  for (int i = 0; i < n_faces; i++) {
    int a, b, c;
    scanf("%d%d%d", &a, &b, &c);
    faces[i] = Face(a, b, c);
    float triangle[3][3];
    triangle[0][0] = vertices[a].x; triangle[0][1] = vertices[a].y; triangle[0][2] = vertices[a].z;
    triangle[1][0] = vertices[b].x; triangle[1][1] = vertices[b].y; triangle[1][2] = vertices[b].z;
    triangle[2][0] = vertices[c].x; triangle[2][1] = vertices[c].y; triangle[2][2] = vertices[c].z;
    root->insert(triangle, i);
  }

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
        double sum = 0;
        for (int t = 0; t < 10; t++) {
          Point P = randomPointTriangle(A, B, C);
          sum += PointMeshDis(P, 2 * average_nn_dis, root);
        }
        candidates[n_candidates] = Face(a, b, c);
        candidates[n_candidates].d = sum / 10;
        n_candidates++;
      }
  }

  freopen(candidates_file.c_str(), "w", stdout);
  printf("%d\n", n_candidates);
  for (int i = 0; i < n_candidates; i++) {
    Face f = candidates[i];
    printf("%d %d %d %lf\n", f.a, f.b, f.c, f.d);  
  }
  return 0;
}