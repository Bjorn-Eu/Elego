#include "node.h"
#include <iostream>
#include <cmath>

double Node::branch_Q(Point move){
    Branch branch = branches.at(move);
    if(branch.visits == 0){
        return 0;
    }
    return branch.total_value/branch.visits;
}
double Node::branch_U(Point move){
    Branch branch = branches.at(move);
    double c = 1.1;
    return c*branch.prior*sqrt(visits)/(1+branch.visits);
}
double Node::branch_score(Point move){
    return branch_Q(move) + branch_U(move);
}


void Node::add_child(Point move, std::shared_ptr<Node> child){
    children.insert({move,child});
}

void Node::update_branch(Point move, double value){
    Branch branch = branches.at(move);
    branch.visits++;
    branch.total_value += value;
    branches.at(move) = branch;
    visits++;
    if (parent != nullptr){
        parent->update_branch(last_move,-value);
    }
}

Point Node::select_branch(){
    double max_score = -10.0;
    Point best_move({0,0});
    for (auto branch: branches){
        double score = branch_score(branch.first);
        if (score > max_score){
            max_score = score;
            best_move = branch.first;
        }
    }
    return best_move;
}

Point Node::select_move(){
    int max_visits = -1.0;
    Point best_move({-1,-1});
    for (auto branch : branches){
        int branch_visits = branch.second.visits;
        if (branch_visits > max_visits){
            max_visits = branch_visits;
            best_move = branch.first;
        }
    }
    return best_move;
}

void Node::print_scores(){
    for (auto branch: branches){
        std::cout<<branch.first.first<<","<<branch.first.second<<": "
        <<branch_Q(branch.first)<<" + "<<branch_U(branch.first)<<" = "<<branch_score(branch.first)<<"\n";
    }
}
void Node::print_visits(){
    for (auto branch: branches){
        std::cout<<branch.first.first<<","<<branch.first.second<<": "<<branch.second.visits<<"\n";
    }
}