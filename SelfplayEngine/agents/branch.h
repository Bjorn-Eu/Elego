#ifndef BRANCH_H
#define BRANCH_H

struct Branch{
    double prior;
    int visits;
    double total_value;
    Branch(double p):prior(p){
        visits = 0;
        total_value = 0;
    }
};

#endif