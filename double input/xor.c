#include<stdio.h>
#include<stdlib.h>
#include<math.h>

typedef struct {
 float or_w1;
 float or_w2;
 float or_b;
 float nand_w1;
 float nand_w2;
 float nand_b;
 float and_w1;
 float and_w2;
 float and_b;
}Xor;  //parameters

float sigmoidf(float x){
    return 1.f / (1.f + expf(-x));
}

float forward(Xor m, float x1, float x2){
    float a = sigmoidf(m.or_w1*x1 + m.or_w2*x2 + m.or_b);
    float b = sigmoidf(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b);

    return sigmoidf(a*m.and_w1 + b*m.and_w2 + m.and_b);
}

typedef float sample[3];

//XOR-agate
sample xor_train[][3] = {  //training data
    {0,0,0},
    {1,0,1},
    {0,1,1},
    {1,1,0},
};

sample *train = xor_train; //switching easy
#define train_count 4

float cost(Xor m){
    float result = 0.0f;
    for(size_t i = 0; i < train_count; i++){
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(m,x1,x2);
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

Xor rand_xor(){
 Xor m;

 float random = rand_float();
 m.or_w1 = rand_float();
 m.or_w2 = rand_float();
 m.or_b = rand_float();
 m.nand_w1 = rand_float();
 m.nand_w2 = rand_float();
 m.nand_b = rand_float();
 m.and_w1 = rand_float();
 m.and_w2 = rand_float();
 m.and_b = rand_float();

 return m;
}

void print_xor(Xor m){
 printf("or_w1 = %f\n", m.or_w1);
 printf("or_w2 = %f\n", m.or_w2);
 printf("or_b = %f\n", m.or_b);
 printf("nand_w1 = %f\n", m.nand_w1);
 printf("nand_w2 = %f\n", m.nand_w2);
 printf("nand_b = %f\n", m.nand_b);
 printf("and_w1 = %f\n", m.and_w1);
 printf("and_w2 = %f\n", m.and_w2);
 printf("and_b = %f\n", m.and_b);
}

float eps = 1e-1;

Xor learn(Xor m, Xor g, float rate){
    m.or_w1 -= rate*g.or_w1;
    m.or_w2 -= rate*g.or_w2;
    m.or_b -= rate*g.or_b;
    m.nand_w1 -= rate*g.nand_w1;
    m.nand_w2 -= rate*g.nand_w2;
    m.nand_b -= rate*g.nand_b;
    m.and_w1 -= rate*g.and_w1;
    m.and_w2 -= rate*g.and_w2;
    m.and_b -= rate*g.and_b;
    return m;
}

Xor finite_diff(Xor m)
{
    Xor g;
    float c = cost(m);
    float saved;

    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c)/eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m) - c)/eps;
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m) - c)/eps;
    m.or_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m) - c)/eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m) - c)/eps;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m) - c)/eps;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m) - c)/eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m) - c)/eps;
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m) - c)/eps;
    m.and_b = saved;

    return g;
}

#define IMG_WIDTH 800
#define IMG_HEIGHT 600

uint32_t img_pixels[IMG_WIDTH*IMG_HEIGHT];

void nn_render(Olivec_Canvas img, NN nn){
    uint32_t low_color = 0x00FF00FF;
    uint32_t high_color = 0x0000FF00;
    
    uint32_t background_color = 0xFF181818;

    olivec_fill(img, background_color);

    int neuron_radius = 25;
    int layer_border_hpad = 50;
    int layer_border_vpad = 50;
    int nn_width = img.width - 2*layer_border_hpad;
    int nn_height = img.height - 2*layer_border_vpad;
    int nn_x = img.width/2 - nn_width/2;
    int nn_y = img.height/2 - nn_height/2;
    size_t arch_count = nn.count + 1;
    int layer_hpad = nn_width / arch_count;

    for(size_t l = 0; l < arch_count; ++l){
        int layer_vpad1 = nn_height/nn.as[l].cols;
        for(size_t i = 0; i < nn.as[l].cols; ++i) {
            int cx1 = nn_x + l*layer_hpad + layer_hpad/2;
            int cy1 = nn_y + i*layer_vpad1 + layer_vpad1/2;
            if(l+1 < arch_count){
                int layer_vpad2 = nn_height / nn.as[l+1].cols;
                for(size_t j = 0; j < nn.as[l+1].cols; ++j){
                    int cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2;
                    int cy2 = nn_y + j*layer_vpad2 + layer_vpad2/2;
                    uint32_t connection_color = 0xFF000000|low_color;
                    uint32_t alpha = floorf(255.f*sigmoidf(MAT_AT(nn.ws[l], i, j)));
                    olivec_blend_color(&connection_color, alpha<<(8*3)|high_color) ;
                    olivec_line(img, cx1, cy1, cx2, cy2, connection_color);
                }
            }
            if(l == 0){
            uint32_t neuron_color = 0xFFAAAAAA;
            olivec_circle(img, cx1, cy1, neuron_radius, neuron_color);
            }
            if(l > 0){
                uint32_t neuron_color = 0xFF000000|low_color;
                uint32_t alpha = floorf(255.f*sigmoidf(MAT_AT(nn.bs[l-1], 0, i)));
                olivec_blend_color(&neuron_color, alpha<<(8*3)|high_color) ;
                olivec_circle(img, cx1, cy1, neuron_radius, neuron_color);
            }
        }
    }
}


int main()
{
    float rate = 1e-1;
    Xor m = rand_xor();

    print_xor(m);
    printf("----------------------\n");

    for(size_t i = 0; i < 500*1000; i++){
        Xor g = finite_diff(m);
        m = learn(m,g,rate);
        // printf("i = %i\n", i);
    }
        printf("cost = %f\n", cost(m));

    printf("-----------------------------\n");

    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            printf("%zu ^ %zu = %f\n", i, j, forward(m,i,j));
        }
    }

    return 0;
}