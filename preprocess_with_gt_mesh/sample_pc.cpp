#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#define N 5555555
using namespace std;

int n_vertices;
int n_faces;

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

}vertices[N];

struct Face {
  int a, b, c;
  Face() {};
  Face (int _a, int _b, int _c) {
    a = _a; b = _b; c = _c;
  };
}faces[N];

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
  string input_file = argv[1];
  string output_file = argv[2];
  int num_l = atoi(argv[3]);
  int num_r = atoi(argv[4]);

  freopen(input_file.c_str(), "r", stdin);
  scanf("%d%d", &n_vertices, &n_faces);

  for (int i = 0; i < n_vertices; i++) {
    double x, y, z;
    scanf("%lf%lf%lf", &x, &y, &z);
    vertices[i] = Point(x, y, z);
  }

  for (int i = 0; i < n_faces; i++) {
    int a, b, c;
    scanf("%d%d%d", &a, &b, &c);
    faces[i] = Face(a, b, c);
  }

  vector<Point> sampled_points;
  double ll = 0.0001, rr = 0.03;
  int iter = 0;
  while (true) {
    double r = (ll + rr) / 2;
    for (int i = 0; i < n_faces; i++) {
      Point A = vertices[faces[i].a];
      Point B = vertices[faces[i].b];
      Point C = vertices[faces[i].c];
      while (true) {
        bool flag = false;
        for (int j = 0; j < 100; j++) {
          Point u = randomPointTriangle(A, B, C);
          bool intersect = false;
          for (Point v: sampled_points) 
            if ((u - v).length() < r) {
              intersect = true;
              break;
            }
          if (!intersect) {
            flag = true;
            sampled_points.push_back(u);
          }
        }
        if (!flag) break;
      }
      if (sampled_points.size() > num_r) 
        break;
    }
    printf("  [iteration: %d, radius: %.5lf, #points: %d]\n", iter, r, sampled_points.size());
    if (sampled_points.size() >= num_l && sampled_points.size() <= num_r)
      break;
    else if (sampled_points.size() < num_l)
      rr = r;
    else
      ll = r;
    sampled_points.clear();
    iter++;
  }

  freopen(output_file.c_str(), "w", stdout);
  printf("%d\n", sampled_points.size());
  for (Point u: sampled_points) 
    printf("%lf %lf %lf\n", u.x, u.y, u.z);
  return 0;
}