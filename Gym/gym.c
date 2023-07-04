// Gym is a GUI app that trains your NN on the data you give it.
// The idea is that it will spit out a binary file that can be then loaded up with nn.h and used in your application

#include "nn.h"

#include "./software/header/raylib.h"

//size of window
#define IMG_FACTOR 80
#define IMG_WIDTH (16*IMG_FACTOR)
#define IMG_HEIGHT (9*IMG_FACTOR)

int main(int argc, char **argv) 
{
    unsigned int buffer_len = 0;
    unsigned char *buffer = LoadFileData( "adder.arch", &buffer_len);    //Load file data as buffer array


    InitWindow(IMG_WIDTH, IMG_HEIGHT, "gym");
    SetTargetFPS(60);

    return 0;
}