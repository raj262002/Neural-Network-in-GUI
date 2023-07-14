// Gym is a GUI app that trains your NN on the data you give it.
// The idea is that it will spit out a binary file that can be then loaded up with nn.h and used in your application

#define NN_IMPLEMENTATION
#include "../framework_in_c/nn.h"
#include<stdio.h>
#include<assert.h>
#include "../software/header/raylib.h"
#define SV_IMPLEMENTATION
#include "sv.h"

//size of window
#define IMG_FACTOR 80
#define IMG_WIDTH (16*IMG_FACTOR)
#define IMG_HEIGHT (9*IMG_FACTOR)

#define DA_INIT_CAP 256

typedef struct {
    size_t *items;
    size_t count;
    size_t capacity;
} Arch;

#define da_append(da, item) \
do {                         \
    if ((da)->count >= (da)->capacity) { \
        (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity * 2; \
        (da)->items = realloc((da)->items, (da)->capacity * sizeof(*(da)->items)); \
        assert((da)->items != NULL && "Buy more RAM lol"); \
    } \
    (da)->items[(da)->count++] = item; \
} while (0)     

char *args_shift(int *argc, char ***argv) {
    assert(*argc > 0);
    // (*argc)--;
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}


int main(int argc, char **argv) 
{
    const char *program = args_shift(&argc, &argv);

    if(argc <= 0) {
        fprintf(stderr, "Usage: %s <model.arch> <model.mat>\n", program);
        fprintf(stderr, "ERROR: no architecture file was provided\n");
        return 1;
    }
    const char *arch_file_path = args_shift(&argc, &argv);

    if(argc <= 0) {
        fprintf(stderr, "Usage: %s <model.arch> <model.mat>\n", program);
        fprintf(stderr, "ERROR: no data file was provided\n");
        return 1;
    }
    const char *data_file_path = args_shift(&argc, &argv);

    unsigned int buffer_len = 0;
    unsigned char *buffer = LoadFileData(arch_file_path, &buffer_len);    //Load file data as buffer array
    if(buffer == NULL) {
        return 1;
    }

    String_View content = sv_from_parts((const char*)buffer, buffer_len);

    Arch arch = {0};

    printf("------------------------\n");
    content = sv_trim_left(content);
    while (content.count > 0 && isdigit(content.data[0])) {
        size_t x = sv_chop_u64(&content);
        da_append(&arch, x);
        content = sv_trim_left(content);
    }
    printf("------------------------\n");

    FILE *in = fopen(data_file_path, "rb");
    if(in == NULL) {
        fprintf(stderr, "ERROR: could not read file %s\n", data_file_path);
        return 1;
    }
    Mat t = mat_load(in);
    fclose(in);

    //TODO : can we have NN with just input?
    NN_ASSERT(arch.count > 1);
    size_t ins_sz = arch.items[0];
    size_t outs_sz = arch.items[arch.count - 1];
    NN_ASSERT(t.cols == ins_sz + outs_sz);

    Mat ti = {
        .rows = t.rows,
        .cols = ins_sz,
        .stride = t.stride,
        .es = &MAT_AT(t,0,0),
   };

    Mat to = {
        .rows = t.rows,
        .cols = outs_sz,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, ins_sz),
    };

    
    MAT_PRINT(ti);
    
    MAT_PRINT(to);

    // InitWindow(IMG_WIDTH, IMG_HEIGHT, "gym");
    // SetTargetFPS(60);

    // size_t i = 0;
    // while(!WindowShouldClose()){
        
    // }

    // printf("hello");
    getchar(); 
    return 0;
}