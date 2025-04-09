#include "mcts.h"
#include "node.h"
#include <iostream>

Point MCTS::select_move(GameState gamestate){
    //create root node
    std::tuple<double, std::map<Point,double>> out = evaluator.evaluate(gamestate);
    std::shared_ptr<Node> root = std::make_shared<Node>(gamestate,nullptr,std::get<1>(out));
    //expand
    for(int i  = 0; i < playouts; i++){
        expand_node(root);
    }
    //std::cout<<"Number of visits: "<<root->visits<<"\n";
    
    //root->print_visits();
    //root->print_scores();
    //root->gamestate.print_gamestate();
    //std::cout<<"Number of legal moves: "<<root->gamestate.legal_moves().size()<<"\n";
    for(auto child : root->children){
        //child.second->gamestate.print_gamestate();
        //std::cout<<"Number of legal moves: "<<child.second->gamestate.legal_moves().size()<<"\n";
        //std::cout<<"Visits: "<<child.second->visits<<"\n";
        //std::cout<<"Number of children: "<<child.second->children.size()<<"\n";
        //std::cout<<"Branch visits: "<<root->branches.at(child.first).visits<<"\n";
    }
    return root->select_move();
}



void MCTS::expand_node(std::shared_ptr<Node> node){
    if(node->gamestate.is_terminal() or node->gamestate.legal_moves().size() == 0){
        //std::cout<<"Game is over\n";
        std::shared_ptr<Node> parent = node->parent;
        parent->update_branch(node->last_move,1.0); //node has been captured
        return;
    }
    Point move = node->select_branch();
    //walk down tree
    if(node->children.count(move)){
        //std::cout<<"Expanding already explored child\n";
        expand_node(node->children.at(move));
    }
    else{
        //std::cout<<"Expanding new child\n";
        GameState new_gamestate = node->gamestate;
        new_gamestate.play_move(move);
        std::tuple<double, std::map<Point,double>> out = evaluator.evaluate(new_gamestate);
        std::shared_ptr<Node> child = std::make_shared<Node>(new_gamestate,node,std::get<1>(out));
        double value = std::get<0>(out);
        child->set_last_move(move);
        node->add_child(move,child);
        node->update_branch(move,-value);
    }
}


