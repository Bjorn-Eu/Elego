#ifndef BOARD_H
#define BOARD_H
#include <vector>
#include <utility>
#include "move.h"
#include <set>
#include <memory>
#include "group.h"
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>

typedef std::unordered_map<Point,std::unordered_set<Point>> AdjacencyTable;

class Board{
    private:      
        void create_adjacency_table();

    public:
        int size;
        std::shared_ptr<AdjacencyTable> adjacency_table;
        std::unordered_map<Point,std::shared_ptr<Group>> groups;

        Board(int s):size(s){        
            adjacency_table = std::make_shared<AdjacencyTable>();
            //groups = std::make_shared<std::unordered_map<Point,std::shared_ptr<Group>>>(); 
            create_adjacency_table();
        }

        Board(int s, std::shared_ptr<AdjacencyTable> at):size(s), adjacency_table(at){
            //groups = std::make_shared<std::unordered_map<Point,std::shared_ptr<Group>>>(); 
        }

        //copy constructor
        Board(const Board& b){
            //std::cout<<"Copy constructor called\n";
            size = b.size;
            adjacency_table = b.adjacency_table;
            for (auto group: b.groups){
                groups.insert({group.first,std::make_shared<Group>(*group.second)});
            }
        }

        //copy assignment
        Board& operator=(const Board& b){
            //std::cout<<"Copy assignment called\n";
            throw std::invalid_argument("Oops someone hasn't implemented this\n");
        }


        bool is_legal(Move move);
        void play_move(Move move);
        bool is_capture(Move move);
        bool is_self_capture(Move move);
        std::vector<Point> legal_moves(int color);
        void print_board();
};
#endif