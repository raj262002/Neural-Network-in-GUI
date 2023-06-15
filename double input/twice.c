#include<stdio.h>
#include<stdlib.h>
#include<time.h>

float train[][2] = {  //training data
    {0,0},
    {1,2},
    {2,4},
    {3,6},
    {4,8},
};

#define train_count sizeof(train)/sizeof(train[0])

// y = x*w; x->input
//GPT-4 -> 1 000 000 000 000 parameters our has 2

float rand_float(void)
{
    return (float) rand()/ (float) RAND_MAX;
}

float cost(float w){
    float result = 0.0f;
    for(size_t i = 0; i < train_count; i++){
        float x = train[i][0];
        float y = x*w ;
        //measuring how well model works
        float d = y - train[i][1];
        result += d*d;
        // printf("actual: %f, expected: %f\n", y, train[i][1]);
    }
    result /= train_count; //calculating average if = 0 prefectly trained
    return result;
}

float dcost(float w){
    float result = 0.0f;
    size_t n = train_count;
    for(size_t i = 0; i < n; ++i){
        float x = train[i][0];
        float y = train[i][1];
        result += 2*(x*w - y)*x;
    }
    result = result / n;
    return result;
}

int main(){

    srand(69);
    // srand(time(0));
    //y = x*w;

    float w0 = rand_float()*10.0f;
    float w = rand_float()*10.0f;
    // float b = rand_float()*5.0f;
    // float w = 1;
    printf("%f\n",w);

    //we know nothing about the data set or traine model right now just guessing

    // float eps = 1e-3;
    float rate = 1e-1;
    // printf("%f\n",cost(w));
    // printf("%f\n",cost(w + eps));
    // printf("%f\n",cost(w + eps*2));

    //if i calculate the derative of cost function i will get the direction where the function grows and i have to move to the negative or opposite

    printf("%f\n",cost(w));
    printf("cost = %f, w = %f\n", cost(w), w);
    for(size_t i = 0; i < 10; ++i){
        // float c = cost(w);
        float dw = dcost(w);
        // float db = (cost(w , b + eps) - c)/eps;
        // printf("%f\n",dcost);
        w -= rate*dw;
        // b -= rate*db;
        printf("cost = %f, w = %f\n", cost(w), w);
    }

    printf("-----------------------------------------\n");
    printf("w = %f\n",w);
    // printf("Hello, Seaman!");
    return 0;
}