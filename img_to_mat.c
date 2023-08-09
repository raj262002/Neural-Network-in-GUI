#include<stdio.h>
#include<assert.h>
#include<float.h>
#include ".\software\header\raylib.h"

#include "stb_image.h"
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#include "./framework_in_c/nn.h"

char *args_shift(int *argc, char ***argv) {
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
} //argument shifting function given by command line

typedef struct {
    size_t *items;
    size_t count;
    size_t capacity;
} Cost_Plot;


#define DA_INIT_CAP 256
#define da_append(da, item) \
do {                         \
    if ((da)->count >= (da)->capacity) { \
        (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity * 2; \
        (da)->items = realloc((da)->items, (da)->capacity * sizeof(*(da)->items)); \
        assert((da)->items != NULL && "Buy more RAM lol"); \
    } \
    (da)->items[(da)->count++] = item; \
} while (0)     

void nn_render_raylib(NN nn, int rx, int ry, int rw, int rh) {
    Color low_color = {0xFF, 0x00, 0xFF, 0xFF};
    Color high_color = {0x00, 0xFF, 0x00, 0xFF};

    // olivec_fill(img, background_color);

    float neuron_radius = rh*0.04;
    int layer_border_hpad = 50;
    int layer_border_vpad = 50;
    int nn_width = rw - 2*layer_border_hpad;
    int nn_height = rh - 2*layer_border_vpad;
    int nn_x = rx + rw/2 - nn_width/2;
    int nn_y = ry + rh/2 - nn_height/2;
    size_t arch_count = nn.count + 1;
    int layer_hpad = nn_width / arch_count;

    for(size_t l = 0; l < arch_count; ++l){
        int layer_vpad1 = nn_height/nn.as[l].cols;
        for(size_t i = 0; i < nn.as[l].cols; ++i) {
            int cx1 = nn_x + l*layer_hpad + layer_hpad/2;
            int cy1 = nn_y + i*layer_vpad1 + layer_vpad1/2;
            if(l+1 < arch_count) {
                int layer_vpad2 = nn_height / nn.as[l+1].cols;
                for(size_t j = 0; j < nn.as[l+1].cols; ++j) {
                    int cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2;
                    int cy2 = nn_y + j*layer_vpad2 + layer_vpad2/2;
                    // uint32_t connection_color = 0xFF000000|low_color;
                    float value = sigmoidf(MAT_AT(nn.ws[l], j, i));
                    high_color.a = floorf(255.f*sigmoidf(MAT_AT(nn.ws[l], i, j)));
                    ColorAlphaBlend(low_color, high_color, WHITE);
                    float thick = rh*0.004f;
                    Vector2 start = {cx1, cy1};
                    Vector2 end   = {cx2, cy2};
                    DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE));
                }
            }
            if(l > 0) {
                high_color.a = floorf(255.f*sigmoidf(MAT_AT(nn.bs[l-1], 0, i)));
                DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));
            }
            else {
                DrawCircle(cx1, cy1, neuron_radius, GRAY);
            }
        }
    }
}

void cost_plot_minmax(Cost_Plot plot, float *min, float *max) {
    *min = FLT_MAX;
    *max = FLT_MIN;
    for(size_t i = 0; i < plot.count; ++i) {
        if(*max < plot.items[i]) *max = plot.items[i];
        if(*min > plot.items[i]) *min = plot.items[i];
    }
}

void plot_cost(Cost_Plot plot, int rx, int ry, int rw, int rh) {
    float min, max;
    cost_plot_minmax(plot, &min, &max);
    for(size_t i = 0; i + 1 < plot.count; ++i) {
        float x1 = rx + (float)rw/plot.count*i;
        float y1 = ry + (1 - (plot.items[i] - min)/(max - min))*rh;
        float x2 = rx + (float)rw/plot.count*(i + 1);
        float y2 = ry + (1 - (plot.items[i + 1] - min)/(max - min))*rh;
        DrawLineEx( (Vector2){x1, y1}, (Vector2){x2, y2}, rh*0.005, RED);
    }
}

void mat_shuffle_rows(Mat m) {
    //shuffling the array algorithm
    //Fisher-Yates shuffle
    for(size_t i = 0; i < m.rows; ++i) {
        size_t j = i + rand()%(m.rows - i);
        if (i != j) {
            for(size_t k = 0; k < m.cols; ++k) {
                float t = MAT_AT(m, i, k);
                MAT_AT(m, i, k) = MAT_AT(m, j, k);
                MAT_AT(m, j, k) = t;
            }
        }
    }
}

int main(int argc, char **argv) 
{
    const char * program = args_shift(&argc, &argv);

    if(argc <= 0) {
        fprintf(stderr, "Usage: %s <input> \n", program);
        fprintf(stderr, "ERROR: no input file is provided\n");
        return 1;
    }

    const char *img_file_path = args_shift(&argc, &argv);

    int img_width, img_height, img_comp;
    uint8_t *img_pixels = stbi_load(img_file_path, &img_width, &img_height, &img_comp, 0);

    if(img_pixels == NULL) {
        fprintf(stderr, "ERROR: could not read image %s\n", img_file_path);
        return 1;
    }
    if(img_comp != 1) {
        fprintf(stderr, "%s is %d bits image. Only 8 bit grayscale images are supported\n", img_file_path, img_comp*8);
        return 1;
    }
    printf("%s size %dx%d %d bits\n", img_file_path, img_width, img_height, img_comp*8);

    Mat t = mat_alloc(img_width*img_height, 3);

    for(int y = 0; y < img_height; ++y) {
        for(int x = 0; x < img_width; ++x) {
            size_t i = y*img_width + x;
            MAT_AT(t, i, 0) = (float)x/(img_width - 1);
            MAT_AT(t, i, 1) = (float)y/(img_height - 1);
            MAT_AT(t, i, 2) = img_pixels[i]/255.f;
        }
    }
    
    mat_shuffle_rows(t);
    // MAT_PRINT(t);
    // return 0;

    // Mat ti = {
    //     .rows = t.rows,
    //     .cols = 2,
    //     .stride = t.stride,
    //     .es = &MAT_AT(t, 0, 0)
    // };
    // Mat to = {
    //     .rows = t.rows,
    //     .cols = 1,
    //     .stride = t.stride,
    //     .es = &MAT_AT(t, 0, ti.cols)
    // };

    // MAT_PRINT(ti);
    // MAT_PRINT(to);

    size_t arch[] = {2, 7, 7, 1};
    NN nn = nn_alloc(arch, ARRAY_LENS(arch));
    NN g = nn_alloc(arch, ARRAY_LENS(arch));
    nn_rand(nn, -1, 1);

    size_t IMG_FACTOR = 80;
    size_t IMG_WIDTH  = (16*IMG_FACTOR);
    size_t IMG_HEIGHT = (9*IMG_FACTOR);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(IMG_WIDTH, IMG_HEIGHT, "gym");
    SetTargetFPS(60);

    Cost_Plot plot = {0};

    Image preview_image = GenImageColor(img_width, img_height, BLACK);
    Texture2D preview_texture = LoadTextureFromImage(preview_image);

    size_t epoch = 0;
    size_t max_epoch = 100*1000;
    size_t batches_per_frame = 100;
    size_t batch_size = 28;
    size_t batch_begin = 0;
    size_t batch_count = (t.rows + batch_size - 1)/batch_size;
    float average_cost = 0;
    float paused = true;
    float rate = 1.0f;

    while(!WindowShouldClose()){
        if(IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            nn_rand(nn, -1, 1);
            plot.count = 0;
        }

        for (size_t i = 0; i < batches_per_frame && !paused && epoch < max_epoch; ++i) {
            size_t size = batch_size;
            if((batch_begin + batch_size) >= t.rows) {
                size = t.rows - batch_begin;
            }

            Mat batch_ti = {
                .rows = size,
                .cols = 2,
                .stride = t.stride,
                .es = &MAT_AT(t, batch_begin, 0),
            };

            Mat batch_to = {
                .rows = size,
                .cols = 1,
                .stride = t.stride,
                .es = &MAT_AT(t, batch_begin, batch_ti.cols),
            };

            nn_backprop(nn, g, batch_ti, batch_to);
            nn_learn(nn, g, rate);
            average_cost += nn_cost(nn, batch_ti, batch_to);
            batch_begin += batch_size;

            if (batch_begin >= t.rows) {
                epoch += 1;
                da_append(&plot, average_cost/batch_count);
                average_cost = 0.0f;
                batch_begin = 0;
            }
        }

        BeginDrawing();
        Color background_color = {0x18, 0x18, 0x18, 0xFF};
        ClearBackground(background_color);
        {
            int w = GetRenderWidth();
            int h = GetRenderHeight();

            int rw = w/3;
            int rh = h*2/3;
            int rx = 0;
            int ry = h/2 - rh/2;

            plot_cost(plot, rx, ry, rw, rh);
            rx += rw;
            nn_render_raylib(nn, rx, ry, rw, rh);
            rx += rw;

            float scale = 10;

            for(size_t y = 0; y < img_height; ++y) {
                for(size_t x = 0; x < img_width; ++x){
                    MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(img_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(img_height - 1);
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
                    ImageDrawPixel(&preview_image, x, y, CLITERAL(Color) {pixel, pixel, pixel, 255});
                }
            printf("\n");
            }

            UpdateTexture(preview_texture, preview_image.data);
            DrawTextureEx(preview_texture, CLITERAL(Vector2) {rx, ry + img_height*scale}, 0, scale, WHITE);

            const char buffer[256];
            snprintf(buffer, sizeof(buffer), "Epoch : %zu/%zu, Rate: %f, Cost: %f", epoch, max_epoch, rate, plot.count > 0 ? plot.items[plot.count - 1] : 0);
            DrawText(buffer, 0, 0, h*0.04, WHITE);
        }
        EndDrawing();
    }

    for(size_t y = 0; y < img_height; ++y) {
        for(size_t x = 0; x < img_width; ++x){
            uint8_t pixel = img_pixels[y*img_width + x];
            if(pixel)
                printf("%3u", pixel);
            else
                printf("  ");
        }
        printf("\n");
    }

    printf("\n");
    printf("-------------------------------\n");
    printf("\n");

    for(size_t y = 0; y < img_height; ++y) {
        for(size_t x = 0; x < img_width; ++x){
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(img_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(img_height - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
            if(pixel)
                printf("%3u", pixel);
            else
                printf("  ");
        }
        printf("\n");
    }

    size_t out_width = 512;
    size_t out_height = 512;
    uint8_t *out_pixels = malloc(sizeof(*out_pixels)*out_width*out_height);
    assert(out_pixels != NULL);


    for(size_t y = 0; y < out_height; ++y) {
    for(size_t x = 0; x < out_width; ++x) {
        MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(out_width - 1);
        MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(out_height - 1);
        nn_forward(nn);
        uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
        out_pixels[y*out_width + x] = pixel;
    }
   }

   const char * out_file_path = "upscaled.png";

   if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width*sizeof(*out_pixels))) {
    fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
    return 1;
   }

   printf("Generated %s from %s\n", out_file_path, img_file_path);
   getchar();

    return 0;
}