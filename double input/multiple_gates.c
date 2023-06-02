#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

float sigmoidf(float x){
    return 1.f / (1.f + expf(-x));
}

typedef float sample[3];

//OR-agate
sample or_train[][3] = {  //training data
    {0,0,0},
    {1,0,1},
    {0,1,1},
    {1,1,1},
};

sample and_train[][3] = {  //training data
    {0,0,0},
    {1,0,0},
    {0,1,0},
    {1,1,1},
};

sample nand_train[][3] = {  //training data
    {0,0,1},
    {1,0,1},
    {0,1,1},
    {1,1,0},
};

//XOR-gate
// (x|y) & ~(x&y)
sample *train = or_train; //switching easy

#define train_count 4

float cost(float w1, float w2, float b){
    float result = 0.0f;
    for(size_t i = 0; i < train_count; i++){
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(x1*w1 + x2*w2 + b);
        //measuring how well model works
        float d = y - train[i][2]; //deviation actual answer - output answer
        result += d*d;
        // printf("actual: %f, expected: %f\n", y, train[i][1]);
    }
    result /= train_count; //calculating average if = 0 prefectly trained
    return result;
}


float rand_float(void)
{
    return (float) rand()/ (float) RAND_MAX;
}


int main()
{
    // srand(69);
    srand(time(0));
    float w0 = rand_float();
    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();
    
    printf("w1 = %f, w2 = %f\n", w1, w2);
    float c = cost(w1,w2,b);
    printf("c = %f\n", c);
    //finding  deratives
    float eps = 1e-2;
    float rate = 1e-2;
    for(size_t i = 0; i < 1000*1000; ++i)
    {
        float c = cost(w1,w2,b);
        // printf("w1 = %f, w2 = %f, c = %f\n",w1,w2,c);
        float dw1 = (cost(w1 + eps,w2,b) - c)/eps;
        float dw2 = (cost(w1 ,w2 + eps,b) - c)/eps;
        float db = (cost(w1 ,w2 ,b + eps) - c)/eps;
        w1 -= rate*dw1;
        w2 -= rate*dw2;
        b -= rate*db;
    }

    printf("w1 = %f, w2 = %f, c = %f, b = %f\n",w1,w2,cost(w1,w2,b), b);

    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 2; ++j){
            printf("%zu | %zu = %f\n", i, j, sigmoidf(i*w1 + j*w2 + b));
        }
    }
    
    return 0;
}