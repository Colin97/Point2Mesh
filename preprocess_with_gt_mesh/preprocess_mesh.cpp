#include <iostream>
#include "octree.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <map>
#include <tuple>
#include <algorithm>
#define N 5555555
using namespace std;

int n_vertices;
int n_faces;
Octree *root;
bool face_valid[N];
int vertex_mapping[N];
int vertex_id[N];
int vertex_cnt[N];

struct Point {
  double x, y, z;
  int id;
  double s;
  bool operator < (const Point & rhs) const {
    return s > rhs.s;
  }
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

}vertices[N], vertices1[N];

struct Face {
  int a, b, c;
  Face() {};
  Face (int _a, int _b, int _c) {
    a = _a; b = _b; c = _c;
  };
}faces[N];

map<tuple<int, int, int>, bool> face_visit;

bool face_check(int a, int b, int c) {
  int ver[3];
  ver[0] = a; ver[1] = b; ver[2] = c;
  sort(&ver[0], &ver[3]);
  if (face_visit[make_tuple(ver[0], ver[1], ver[2])])
    return false;
  face_visit[make_tuple(ver[0], ver[1], ver[2])] = true;
  return true;
}

map<pair<int, int>, vector<int> > edge_faces;

void add_edge_faces(int a, int b, int i) {
  if (a > b)
    swap(a, b);
  edge_faces[make_pair(a, b)].push_back(i);
  return ;
}

vector<int> get_edge_faces(int a, int b) {
  if (a > b)
    swap(a, b);
  return edge_faces[make_pair(a, b)];
}

bool point_in_segment(int a, int b, int c) {
  if (c == a || c == b)
    return false;
  Point A = vertices[a];
  Point B = vertices[b];
  Point C = vertices[c];
  double lAB = (B - A).length();
  double t = ((B - A).dot(C - A)) / (lAB * lAB);
  if (t < 0.01 || t > 0.99)
    return false;
  Point D = A + ((B - A) * t);
  return (D - C).length() < 1e-3;
}

bool insert_face(int a, int b, int c, int id) {
  if (a == b || a == c || b == c)
    return false;
  if (!face_check(a, b, c)) 
    return false;
  faces[id] = Face(a, b, c);
  face_valid[id] = true;
  float triangle[3][3];
  triangle[0][0] = vertices[a].x; triangle[0][1] = vertices[a].y; triangle[0][2] = vertices[a].z;
  triangle[1][0] = vertices[b].x; triangle[1][1] = vertices[b].y; triangle[1][2] = vertices[b].z;
  triangle[2][0] = vertices[c].x; triangle[2][1] = vertices[c].y; triangle[2][2] = vertices[c].z;
  root->insert(triangle, id);
  add_edge_faces(a, b, id);
  add_edge_faces(a, c, id);
  add_edge_faces(c, b, id);
  return true;
}

void split(int a, int b, int p) {
  vector<int> face_ids = get_edge_faces(a, b);
  for (int id: face_ids) {
    if (!face_valid[id]) continue;
    Face f = faces[id];
    int c;
    if (f.a != a && f.a != b)
      c = f.a;
    else if (f.b != a && f.b != b)
      c = f.b;
    else
      c = f.c;
    if (p == c)
      continue;
    face_valid[id] = false;
    insert_face(a, p, c, n_faces);
    n_faces++;
    insert_face(b, p, c, n_faces);
    n_faces++;
  }
  return ;
}

int main(int argc, char ** argv) {
  string input_file = argv[1];
  string output_file = argv[2];

  freopen(input_file.c_str(), "r", stdin);
  scanf("%d%d", &n_vertices, &n_faces);
  for (int i = 0; i < n_vertices; i++) {
    double x, y, z;
    scanf("%lf%lf%lf", &x, &y, &z);
    vertices[i] = Point(x, y, z);
    vertex_cnt[i] = 0;
  }

  for (int i = 0; i < n_faces; i++) {
    int a, b, c;
    scanf("%d%d%d", &a, &b, &c);
    faces[i].a = a; faces[i].b = b; faces[i].c = c;
    vertex_cnt[a]++; vertex_cnt[b]++; vertex_cnt[c]++;
  }

  double xmin = 1e9, ymin = 1e9, zmin = 1e9;
  double xmax = -1e9, ymax = -1e9, zmax = -1e9;
  for (int i = 0; i < n_vertices; i++) {
    if (vertex_cnt[i] == 0)
      continue;
    double x = vertices[i].x, y = vertices[i].y, z = vertices[i].z;
    xmax = max(xmax, x); ymax = max(ymax, y); zmax = max(zmax, z);
    xmin = min(xmin, x); ymin = min(ymin, y); zmin = min(zmin, z);
  }

  double scale = sqrt((xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin) + (zmax - zmin) * (zmax - zmin));

  for (int i = 0; i < n_vertices; i++) {
    vertices[i].x = (vertices[i].x - (xmax + xmin) / 2) / scale;
    vertices[i].y = (vertices[i].y - (ymax + ymin) / 2) / scale;
    vertices[i].z = (vertices[i].z - (zmax + zmin) / 2) / scale;
    vertices1[i] = vertices[i];
    vertices1[i].s = 0;
    vertices1[i].id = i;
  }
  
  for (int i = 0; i < n_faces; i++) {
    int a = faces[i].a, b = faces[i].b, c = faces[i].c;
    double s = (vertices[c] - vertices[a]).cross(vertices[b] - vertices[a]).length();
    vertices1[a].s = max(vertices1[a].s, s);
    vertices1[b].s = max(vertices1[b].s, s);
    vertices1[c].s = max(vertices1[c].s, s);
  }

  sort(&vertices1[0], &vertices1[n_vertices]);

  for (int i = 0; i < n_vertices; i++) {
    int uid = vertices1[i].id;
    if (vertex_cnt[uid] == 0) {
      vertex_mapping[uid] = -1;
      continue;
    }
    vertex_mapping[uid] = uid;
    for (int j = 0; j < i; j++) {
      int vid = vertices1[j].id;
      if (vertex_mapping[vid] == vid && (vertices[uid] - vertices[vid]).length() < 0.001) {
        vertex_mapping[uid] = vertex_mapping[vid];
      }
    }
  }

  double range_l[3] = {-0.6, -0.6, -0.6};
  double range_r[3] = {0.6, 0.6, 0.6};
  root = new Octree(range_l, range_r, 0.01);
  for (int i = 0; i < n_faces; i++) {
    Face f = faces[i];
    insert_face(vertex_mapping[f.a], vertex_mapping[f.b], vertex_mapping[f.c], i);
  }

  for (int i = 0; i < n_vertices; i++) {
    if (vertex_mapping[i] != i) continue;
    double query_l[3], query_r[3];

    query_l[0] = vertices[i].x; query_l[1] = vertices[i].y; query_l[2] = vertices[i].z;
    query_r[0] = vertices[i].x; query_r[1] = vertices[i].y; query_r[2] = vertices[i].z;
    set<int> face_id;
    root->query(query_l, query_r, &face_id);
    for (int id: face_id) {
      if (!face_valid[id]) continue;
      Face f = faces[id];
      if (point_in_segment(f.a, f.b, i)) 
        split(f.a, f.b, i);
      else if (point_in_segment(f.a, f.c, i)) 
        split(f.a, f.c, i);
      else if (point_in_segment(f.b, f.c, i)) 
        split(f.b, f.c, i);
    }
  }

  int n_new_vertices = 0, n_new_faces = 0;
  for (int i = 0; i < n_vertices; i++) {
    if (vertex_mapping[i] == i) {
      vertex_id[i] = n_new_vertices;
      n_new_vertices++;
    }
  }
  for (int i = 0; i < n_faces; i++) 
    if (face_valid[i])
      n_new_faces++;
  
  freopen(output_file.c_str(), "w", stdout);
  printf("%d\n%d\n", n_new_vertices, n_new_faces);
  for (int i = 0; i < n_vertices; i++)
    if (vertex_mapping[i] == i)
    printf("%lf %lf %lf\n", vertices[i].x, vertices[i].y, vertices[i].z);
  for (int i = 0; i < n_faces; i++)
    if (face_valid[i])
      printf("%d %d %d\n", vertex_id[faces[i].a], vertex_id[faces[i].b], vertex_id[faces[i].c]);
  return 0;
}