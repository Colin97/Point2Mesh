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

struct Point {
  double x, y, z;
  Point() {};
  Point (double _x, double _y, double _z) {
    x = _x; y = _y; z = _z;
  };
  Point operator - (const Point& v) const {
        return Point(x - v.x, y - v.y, z - v.z);
    };

    Point operator + (const Point& v) const {
        return Point(x + v.x, y + v.y, z + v.z);
    };

    Point operator * (const double t) const {
      return Point(x * t, y * t, z * t);
    }

    double length() {
      return sqrt(x * x + y * y + z * z);
    }

    void normalize() {
      double l = length();
      x /= l; y /= l; z /= l;
      return ;
    }

    double dot(const Point& v) const {
        return x * v.x + y * v.y + z * v.z;
    };

    Point cross(const Point& v) const {
        return Point(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x);};
}vertices[N], pc[N];

struct Face {
  int a, b, c;
  Face() {};
  Face (int _a, int _b, int _c) {
    a = _a; b = _b; c = _c;
  };
}faces[N];

int n_vertices, n_faces, n_pc;
vector<int> face_points[N];
map<pair<int, int>, vector<int> > edge_faces;
vector<int> vertex_faces[N];
int stack_A[N], stack_B[N], stack_C[N], stack_face[N];
Point AA[N], BB[N], CC[N];
double average_nn_dis = 0;
Point S, SS, R1[N], R2[N];
int s_id, start_time;
const int timeout_sec = 60;
map<pair<int, int>, double> geo_dis;
AnnoyIndex<int, double, Euclidean, Kiss32Random> pc_knn = AnnoyIndex<int, double, Euclidean, Kiss32Random>(3);

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

void update_geo_dis(int u, int v, double d) {
  if (u > v)
    swap(u, v);
  if (geo_dis[make_pair(u, v)] < 1e-9 || d < geo_dis[make_pair(u, v)])
    geo_dis[make_pair(u, v)] = d;
  return ;
}

Point compute_CC(Point S, Point A, Point B, Point C, Point AA, Point BB) {
  Point norm = (AA - BB).cross(Point(0, 0, 1));
  norm.normalize();
  double lAB = (A - B).length();
  double d1 = (C - A).dot(B - A) / (lAB * lAB);
  double d2 = (C - A).cross(B - A).length() / lAB;
  Point CC = AA + (BB - AA) * d1 + norm * d2;
  if ((S - AA).cross(BB - AA).dot((BB - AA).cross(CC - AA)) < 0)
      CC = AA + (BB - AA) * d1 - norm * d2;
  return CC;
}

Point compute_PP(Point P, Point A, Point B, Point C, Point AA, Point BB, Point CC) {
  double s0 = (B - P).cross(C - P).length();
  double s1 = (A - P).cross(C - P).length();
  double s2 = (A - P).cross(B - P).length();
  double s = s0 + s1 + s2;
  return AA * (s0 / s) + BB * (s1 / s) + CC * (s2 / s);
}

int dir(Point ray1, Point ray2) {
  double tmp = ray1.cross(ray2).dot(Point(0, 0, 1));
  if (tmp > 1e-9)
    return 1;
  else if (tmp < -1e-9)
    return -1;
  else
    return 0;
}

bool in_two_ray(Point A, Point B, Point C) {
  return (dir(A, C) != -1 && dir(C, B) != -1);
}

void connect(int depth) {
  vector<int> points = face_points[stack_face[depth]];
  for (int q_id: points) {
    Point Q = compute_PP(pc[q_id], vertices[stack_A[depth]], vertices[stack_B[depth]], 
                         vertices[stack_C[depth]], AA[depth], BB[depth], CC[depth]);
    Point R3 = (Q - SS);
    double d = R3.length();
    if (d > 5 * average_nn_dis) 
      continue;
    R3.normalize();
    if (in_two_ray(R1[depth - 1], R2[depth - 1], R3) == false)
      continue;
    update_geo_dis(s_id, q_id, d);
  }
}

bool check_dis(int depth) {
  for (int i = 1; i < 10; i++) {
    Point p = (BB[depth] - AA[depth]) * (0.1 * i) + AA[depth];
    if ((p - SS).length() < 5 * average_nn_dis)
      return true;
  }
  return false;
}

bool update_ray(int depth) {
  Point r1 = AA[depth + 1] - SS;
  Point r2 = BB[depth + 1] - SS;
  r1.normalize();
  r2.normalize();
  if (dir(r1, r2) == -1)
    swap(r1, r2);  
  if (depth == 0) {
    R1[0] = r1;
    R2[0] = r2;
    return true;
  }
  R1[depth] = dir(R1[depth - 1], r1) == 1 ? r1 : R1[depth - 1];
  R2[depth] = dir(r2, R2[depth - 1]) == 1 ? r2 : R2[depth - 1];
  if (dir(R1[depth], R2[depth]) == -1) {
    return false;
  }
  if (!(in_two_ray(R1[depth], R2[depth], r1) ||
        in_two_ray(R1[depth], R2[depth], r2) || 
        in_two_ray(r1, r2, R1[depth]) ||
        in_two_ray(r1, r2, R2[depth])))
    return false;

  if (R1[depth].dot(R2[depth]) > 0.999) 
    return false;
  return true;
}

void unfold_path(int depth) {
  if (clock() - start_time > timeout_sec * CLOCKS_PER_SEC) {
    printf("Error: time out when calculating geo distances.\n");
    exit(0);
  }
  if (depth > 25)
    return ;

  for (int i = 0; i < depth; i++)
    if (stack_face[i] == stack_face[depth])
      return ;

  Point A = vertices[stack_A[depth]];
  Point B = vertices[stack_B[depth]];
  Point C = vertices[stack_C[depth]];
  if (depth == 0) {
    AA[0] = Point(0, 0, 0);
    BB[0] = Point((A - B).length(), 0, 0);
    CC[0] = compute_CC(Point(3, -1, 0), A, B, C, AA[0], BB[0]);
    SS = compute_PP(S, A, B, C, AA[0], BB[0], CC[0]);
  }
  else {
    connect(depth);
    if (!check_dis(depth))
      return ;
  }

  vector<int> adj_face_ids = stack_A[depth] < stack_C[depth] ? 
                             edge_faces[make_pair(stack_A[depth], stack_C[depth])] :
                             edge_faces[make_pair(stack_C[depth], stack_A[depth])];
  stack_A[depth + 1] = stack_A[depth];
  stack_B[depth + 1] = stack_C[depth];
  AA[depth + 1] = AA[depth];
  BB[depth + 1] = CC[depth];
  for (int id: adj_face_ids) 
    if (id != stack_face[depth]) {
      stack_face[depth + 1] = id;
      Face face = faces[id];
      if (face.a != stack_A[depth] && face.a != stack_C[depth])
        stack_C[depth + 1] = face.a;
      else if (face.b != stack_A[depth] && face.b != stack_C[depth])
        stack_C[depth + 1] = face.b;
      else
        stack_C[depth + 1] = face.c;
      CC[depth + 1] = compute_CC(BB[depth], A, C, vertices[stack_C[depth + 1]], AA[depth], CC[depth]);
      if (update_ray(depth))
        unfold_path(depth + 1);
    }

  if (depth == 0)
    return ;
  adj_face_ids = stack_B[depth] < stack_C[depth] ?
                 edge_faces[make_pair(stack_B[depth], stack_C[depth])] :
                 edge_faces[make_pair(stack_C[depth], stack_B[depth])];
  stack_A[depth + 1] = stack_B[depth];
  stack_B[depth + 1] = stack_C[depth];
  AA[depth + 1] = BB[depth];
  BB[depth + 1] = CC[depth];
  for (int id: adj_face_ids) 
    if (id != stack_face[depth]) {
      stack_face[depth + 1] = id;
      Face face = faces[id];
      if (face.a != stack_B[depth] && face.a != stack_C[depth])
        stack_C[depth + 1] = face.a;
      else if (face.b != stack_B[depth] && face.b != stack_C[depth])
        stack_C[depth + 1] = face.b;
      else
        stack_C[depth + 1] = face.c;
      CC[depth + 1] = compute_CC(AA[depth], B, C, vertices[stack_C[depth + 1]], BB[depth], CC[depth]);
      if (update_ray(depth))
        unfold_path(depth + 1);
    }
  return ;
}

int main(int argc, char ** argv) {
  string pc_file = argv[1];
  string mesh_file = argv[2];
  string geo_dis_file = argv[3];

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
    std::vector<int> closest;
    std::vector<double> dis;
    pc_knn.get_nns_by_item(i, 5, -1, &closest, &dis);
    average_nn_dis += dis[1];
  }
  average_nn_dis /= n_pc;

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
    double query_l[3], query_r[3];
    Point p = pc[i];
    query_l[0] = p.x; query_l[1] = p.y; query_l[2] = p.z;
    query_r[0] = p.x; query_r[1] = p.y; query_r[2] = p.z;
    set<int> face_id;
    root->query(query_l, query_r, &face_id);
    double min_dis = 1e9;
    int min_id = -1;
    for (int id: face_id) {
      Face face = faces[id];
      Point A = vertices[face.a], B = vertices[face.b], C = vertices[face.c];
      double dis = disPointTriangle(p, A, B, C);
      if (dis < min_dis) {
        min_dis = dis;
        min_id = id;
      }
    }
    face_points[min_id].push_back(i);
  }

  for (int i = 0; i < n_faces; i++) {
    int ver[3];
    ver[0] = faces[i].a;
    ver[1] = faces[i].b;
    ver[2] = faces[i].c;
    sort(&ver[0], &ver[3]);
    edge_faces[make_pair(ver[0], ver[1])].push_back(i);
    edge_faces[make_pair(ver[0], ver[2])].push_back(i);
    edge_faces[make_pair(ver[1], ver[2])].push_back(i);
    vertex_faces[ver[0]].push_back(i);
    vertex_faces[ver[1]].push_back(i);
    vertex_faces[ver[2]].push_back(i);
  }
  
  start_time = clock();
  for (int i = 0; i < n_faces; i++) {
    if (clock() - start_time > timeout_sec * CLOCKS_PER_SEC) {
      printf("Error: time out when calculating geo distances.\n");
      exit(0);
    }
    int size = face_points[i].size();
    if (size == 0)
      continue;
    for (int j = 0; j < size; j++)
      for (int k = j + 1; k < size; k++) {
        Point U = pc[face_points[i][j]];
        Point V = pc[face_points[i][k]];
        double d = (U - V).length();
        if (d < 5 * average_nn_dis)
          update_geo_dis(face_points[i][j], face_points[i][k], d);
      }
    for (int j = 0; j < size; j++) {
      s_id = face_points[i][j];
      S = pc[s_id];
      int a = faces[i].a, b = faces[i].b, c = faces[i].c;
      stack_face[0] = i;
      stack_A[0] = a; stack_B[0] = b; stack_C[0] = c; 
      unfold_path(0);
      stack_A[0] = c; stack_B[0] = a; stack_C[0] = b;
      unfold_path(0); 
      stack_A[0] = b; stack_B[0] = c; stack_C[0] = a; 
      unfold_path(0);
    }
  }

  for (int i = 0; i < n_vertices; i++) {
    for (int j = 0; j < vertex_faces[i].size(); j++)
      for (int k = j + 1; k < vertex_faces[i].size(); k++) {
        int f0 = vertex_faces[i][j];
        int f1 = vertex_faces[i][k];
        for (int u: face_points[f0])
          for (int v: face_points[f1]) {
            double d = (pc[u] - vertices[i]).length() + (pc[v] - vertices[i]).length();
            if (d < 5 * average_nn_dis)
              update_geo_dis(u, v, d);
          }
      }
  }

  freopen(geo_dis_file.c_str(), "w", stdout);
  int pair_cnt = 0;
  for (auto p: geo_dis)
    pair_cnt++;
  printf("%d\n", pair_cnt);
  for (auto p: geo_dis)
    printf("%d %d %lf\n", p.first.first, p.first.second, p.second);
  return 0;
}