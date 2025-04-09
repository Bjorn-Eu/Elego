#ifndef TEST_POLICY_H
#define TEST_POLICY_H
#include <tuple>


std::tuple<double,std::map<Point,double>> get_pure_out(torch::Tensor x, int size){
    double value = 0;
    std::map<Point,double> policy;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            policy.insert({{i,j},1.0/(size*size)});
        }
    }
    return std::make_tuple(value,policy);
}

std::tuple<double,std::map<Point,double>> get_pol_out(torch::Tensor x, int size){
    double value = 0;
    std::map<Point,double> policy;
    
    torch::Tensor policy_tensor = (torch::zeros({size,size},torch::kFloat32));
    if (x.size(0) != 0){
        auto np_x = x.data_ptr<float>();
        for (int i = 1; i < size-1; i++){
            for (int j = 1; j < size-1; j++){
                int val = 0;
                if (np_x[2*size*size+j*size+i] == 1){
                    val = 1000;
                }
                else if (np_x[3*size*size+j*size+i] == 1){
                    val = 100;
                }
                else if (np_x[4*size*size+j*size+i] == 1){
                    val = 10;
                }
                if (val != 0){
                    policy_tensor[i+1][j] = val;
                    policy_tensor[i-1][j] = val;
                    policy_tensor[i][j+1] = val;
                    policy_tensor[i][j-1] = val;
                }
            }
        }
    }
    policy_tensor = torch::softmax(policy_tensor,0);
    auto np_policy = policy_tensor.data_ptr<float>();
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            policy.insert({{i,j},np_policy[i*size+j]});
        }
    }
    
    return std::make_tuple(value,policy);
}


std::tuple<double,std::map<Point,double>> get_q_out(torch::Tensor x, int size){
    double value = 0;
    std::map<Point,double> policy;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            policy.insert({{i,j},1.0/(size*size)});
        }
    }
    auto np_x = x.data_ptr<float>();
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            if (np_x[2*size*size+j*size+i] == 1 && np_x[1*size*size+j*size+i] == 1){
                value = 1;
                i = size;
                break;
            }
        }
    }
    return std::make_tuple(value,policy);
}


std::tuple<double,std::map<Point,double>> get_adv_out(torch::Tensor x, int size){
    double value = std::get<0>(get_q_out(x,size));
    std::map<Point,double> policy = std::get<1>(get_pol_out(x,size));
    return std::make_tuple(value,policy);
}



#endif