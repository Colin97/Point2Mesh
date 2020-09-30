#ifndef OCTREE_H_
#define OCTREE_H_

#include <vector>
#include <set>
#include "Intersection.h"

class Octree {
public:
	double range_l[3], range_r[3];
	double l_min;
	float box_center[3], box_half_size[3];
	std::set<int> face_ids;
	Octree *children[8];
	bool is_leaf;
	Octree() {}
	Octree(double _range_l[3], double _range_r[3], double _l_min) {
		for (int i = 0; i < 3; i++) {
			range_l[i] = _range_l[i];
			range_r[i] = _range_r[i];
			box_center[i] = (range_l[i] + range_r[i]) / 2;
			box_half_size[i] = (range_r[i] - range_l[i]) / 2;
		}
		l_min = _l_min;
		for (int i = 0; i < 8; i++) children[i] = NULL;
		is_leaf = (range_r[0] - range_l[0]) < l_min && (range_r[1] - range_l[1]) < l_min && (range_r[2] - range_l[2]) < l_min;
	}
	void insert(float triangle[3][3], int id) {
		if (triBoxOverlap(box_center, box_half_size, triangle) == 0)
			return ;
		face_ids.insert(id);
		if (is_leaf)
			return ;
		for (int i = 0; i < 8; i++) {
			if (children[i] == NULL) {
				double _range_l[3], _range_r[3];
				for (int j = 0; j < 3; j++)
					if (i & (1 << j)) {
						_range_l[j] = range_l[j];
						_range_r[j] = (range_l[j] + range_r[j]) / 2;
					}
					else {
						_range_l[j] = (range_l[j] + range_r[j]) / 2;
						_range_r[j] = range_r[j];
					}
				children[i] = new Octree(_range_l, _range_r, l_min);
			}
			children[i]->insert(triangle, id);
		}
	}
	void query(double query_l[3], double query_r[3], std::set <int> *ans) {
		if ((query_l[0] - l_min < range_l[0] && query_r[0] + l_min > range_r[0] && 
		     query_l[1] - l_min < range_l[1] && query_r[1] + l_min > range_r[1] && 
			 query_l[2] - l_min < range_l[2] && query_r[2] + l_min > range_r[2])) {
			for (int face_id: face_ids)
				ans->insert(face_id);
			return ;
		}
		for (int i = 0; i < 8; i++) 
			if (children[i] != NULL) {
				if (query_l[0] - l_min > children[i]->range_r[0] || query_r[0] + l_min < children[i]->range_l[0] || 
					query_l[1] - l_min > children[i]->range_r[1] || query_r[1] + l_min < children[i]->range_l[1] || 
					query_l[2] - l_min > children[i]->range_r[2] || query_r[2] + l_min < children[i]->range_l[2])
					continue;
				children[i]->query(query_l, query_r, ans);
			}
		return ;
	}
};

#endif

