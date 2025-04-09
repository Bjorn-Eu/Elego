#ifndef TEST_NETS_H
#define TEST_NETS_H
#include <torch/torch.h>
#include <tuple>
#include <map>

/*
std::map<Point,double> get_policy_map(torch::Tensor policy_tensor,int size){
    std::map<Point,double> policy;
    auto np_policy = policy_tensor.data_ptr<float>();
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            policy.insert({{i,j},np_policy[i*size+j]});
        }
    }
    return policy;
}*/



struct WNET_PURE : torch::nn::Module {
    int size;
    double scale;
    WNET_PURE(int s):size(s){
        scale = 1/(size*size);
    }

    std::tuple<torch::Tensor,torch::Tensor> forward(torch::Tensor x){
        torch::Tensor policy = scale*(torch::ones({size,size},torch::kFloat32));
        torch::Tensor value = torch::zeros({1},torch::kFloat32);
        return std::make_tuple(value, policy);
    }
};

struct WNET_POL : torch::nn::Module {
    int size;
    double scale;
    WNET_POL(int s):size(s){
        scale = 1/(size*size);
    }
    std::tuple<torch::Tensor,torch::Tensor> forward(torch::Tensor x){
        torch::Tensor policy = scale*(torch::ones({size,size},torch::kFloat32));
        torch::Tensor value = torch::zeros({1},torch::kFloat32);
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
                        policy[i+1][j] = val;
                        policy[i-1][j] = val;
                        policy[i][j+1] = val;
                        policy[i][j-1] = val;
                    }
                }
            }
        }

    return std::make_tuple(value, policy);
    }
};


struct WNET_Q : torch::nn::Module {
    int size;
    double scale;
    WNET_Q(int s):size(s){
        scale = 1/(size*size);
    }
    std::tuple<torch::Tensor,torch::Tensor> forward(torch::Tensor x){
        torch::Tensor policy = scale*(torch::ones({size,size},torch::kFloat32));
        torch::Tensor value = torch::zeros({1},torch::kFloat32);
        for(int i = 0; i < size; i++){
            for (int j = 0; j < size; j++){
                //if((x[2][j][i] == 1) && (x[1][j][i] == 1)){
                    value[0] = 1;
                    break;
                //}
            }
        }

        return std::make_tuple(value, policy);
    }
};

struct WNET_RANDOM : torch::nn::Module {
    int size;
    WNET_RANDOM(int s):size(s){}
    std::tuple<torch::Tensor,torch::Tensor> forward(torch::Tensor x){
        torch::Tensor policy = (torch::rand({size,size},torch::kFloat32));
        torch::Tensor value = 2*torch::rand({1},torch::kFloat32)+1.0;
        return std::make_tuple(value, policy);
    }
};
//Only adjust Q to 1 if there is a capture
//and keeps policy set to uniform
/*
class WNET_Q(nn.Module):
    def __init__(self,size=9):
        super().__init__()
        self.size = size
        self.scale = 1/(self.size*self.size)
    def forward(self,x):
        policy = self.scale*(np.ones((self.size*self.size),dtype=np.float32))        
        policy = torch.from_numpy(policy)

        value = 0
        if x.shape[0] != 0:
            np_x = x.numpy()
            np_x.shape = (5,self.size,self.size)
            for i in range(0,self.size):
                for j in range(0,self.size):
                    if np_x[2][j][i]==1 and np_x[1][j][i]==1:
                        value = 1
                        break

        value = np.array([value])
        value = torch.tensor(value)
        return value, policy

/*
class WNET_ADV(nn.Module):
    def __init__(self,size=9):
        super().__init__()
        self.size = size
        self.scale = 1/(self.size*self.size)

    def forward(self,x):
        policy = self.scale*(np.ones((self.size,self.size),dtype=np.float32))
        value = 0
        if x.shape[0] != 0:
            np_x = x.numpy()
            np_x.shape = (5,self.size,self.size)
            
            for i in range(1,self.size-1):
                for j in range(1,self.size-1):
                    val =  0
                    if np_x[2][j][i] == 1:
                        val = 1000
                        if np_x[1][j][i]==1:
                            value = 1
                    elif np_x[3][j][i] == 1:
                        val = 100
                    elif np_x[4][j][i] == 1:
                        val = 10
                    if val != 0:
                        policy[i+1][j] = val
                        policy[i-1][j] = val
                        policy[i][j+1] = val
                        policy[i][j-1] = val

        policy.shape = (self.size*self.size)
        policy = torch.from_numpy(policy)

        value = np.array([value])
        value = torch.tensor(value)
        return value, policy
*/



#endif