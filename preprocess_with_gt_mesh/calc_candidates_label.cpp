#include <cstdio>
#include <cstdlib>
#include <map>
#include <cmath>
#define N 5555555
using namespace std;

int n_pc;
int n_candidates;
map<pair<int, int>, double> geo_dis;

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
  int a, b, c, type;
  Face() {};
  Face (int _a, int _b, int _c) {
    a = _a; b = _b; c = _c;
  };
}candidates[N * 10];

double get_geo_dis(int u, int v) {
  if (u > v) swap(u, v);
  double d = geo_dis[make_pair(u, v)];
  if (fabs(d) < 1e-9)
    return 1e9;
  return d;
}

int main(int argc, char ** argv) {
  string points_file = argv[1];
  string geo_dis_file = argv[2];
  string candidates_file = argv[3];
  string label_file = argv[4];
  double tau = atof(argv[5]);

  freopen(points_file.c_str(), "r", stdin);
  scanf("%d", &n_pc);
  for (int i = 0; i < n_pc; i++) {
    double vec[3];
    scanf("%lf%lf%lf", &vec[0], &vec[1], &vec[2]);
    pc[i] = Point(vec[0], vec[1], vec[2]);
  }

  freopen(geo_dis_file.c_str(), "r", stdin);
  int n_pairs;
  scanf("%d", &n_pairs);
  for (int i = 0 ; i < n_pairs; i++) {
    int u, v;
    double d;
    scanf("%d%d%lf", &u, &v, &d);
    if (u > v)
      swap(u, v);
    geo_dis[make_pair(u, v)] = d;
  }

  freopen(candidates_file.c_str(), "r", stdin);
  scanf("%d", &n_candidates);
  for (int i = 0; i < n_candidates; i++) {
    int a, b, c;
    double d;
    scanf("%d %d %d %lf", &a, &b, &c, &d);
    Point A = pc[a];
    Point B = pc[b];
    Point C = pc[c];
    double l1 = (A - B).length();
    double l2 = (A - C).length();
    double l3 = (B - C).length();
    double ratio = (get_geo_dis(a, b) + get_geo_dis(a, c) + get_geo_dis(b, c)) / (l1 + l2 + l3);
    candidates[i] = Face(a, b, c);
    if (ratio > tau)
      candidates[i].type = 0;
    else 
        candidates[i].type = d < 0.0005 ? 1 : 2;
  }

  freopen(label_file.c_str(), "w", stdout);
  printf("%d\n", n_candidates);
  for (int i = 0; i < n_candidates; i++) {
    Face f = candidates[i];
    printf("%d %d %d %d\n", f.a, f.b, f.c, f.type);
  }
  return 0;
}