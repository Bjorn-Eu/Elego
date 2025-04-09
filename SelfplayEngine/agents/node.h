#ifndef NODE_H
#define NODE_H
#include <memory>
#include <map>
#include <vector>
#include "../gameplay/gamestate.h"
#include "branch.h"

class Node{
    public:
        GameState gamestate;
        std::shared_ptr<Node> parent;
        int visits = 1;
        std::map<Point,std::shared_ptr<Node>> children;
        std::map<Point, Branch> branches;
        
        Point last_move;
        double branch_Q(Point move);
        double branch_U(Point move);
        double branch_score(Point move);
        void add_child(Point move, std::shared_ptr<Node> child);
        void update_branch(Point move, double value);
        Point select_branch();
        Point select_move();

        void print_scores();
        void print_visits();
        void set_last_move(Point move){
            last_move = move;
        }
        Node(GameState g, std::shared_ptr<Node> p,std::map<Point,double> priors): gamestate(g), parent(p){
            std::vector<Point> moves = gamestate.legal_moves();
            for(auto move : moves){
                branches.insert({move,priors.at(move)});
            }
        }
}; 

#endif