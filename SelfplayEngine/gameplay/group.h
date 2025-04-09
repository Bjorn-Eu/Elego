/*
Encodes a group of stones on the board together with it liberties.
The management of the group and its liberties is done by the board class.
*/
#ifndef GROUP_H
#define GROUP_H

#include <set>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

//typedef std::pair<int,int> Point;
typedef struct Point{
    int first;
    int second;
    bool operator<(const Point& other) const {
        //return std::tie(first, second) < std::tie(other.first, other.second);
        return (9*first + second) < (9*other.first + other.second);
    }
    Point(){}
    Point(int x, int y):first(x), second(y){}

    std::size_t operator()(const std::pair<int, int> &p) const{
        return std::hash<int>{}(9*p.second + p.first);
    }

    bool operator==(const Point &other) const {
        return (first == other.first && second == other.second);
    }
} Point; 


template <>
struct std::hash<Point>
{
  std::size_t operator()(const Point& p) const
  {
    using std::size_t;
    using std::hash;

    return hash<int>()(p.second + 9*p.first);
  }
};

class Group{
    private:

    public:
        int color;
        std::unordered_set<Point> stones; //not to be modified
        std::unordered_set<Point> liberties; //not to be modified

        Group(int c,std::unordered_set<Point> s,std::unordered_set<Point> l)
        : color(c), stones(s), liberties(l){
            //std::cout<<"Group constructor 2 called\n";
        }
        
        int number_of_liberties() const;
        int number_of_stones() const;
        void merge_group(const Group group);
        void remove_liberty(const Point liberty);
};


#endif