#include "./software/header/raylib.h"
#include<stdio.h>

int main()
{
    const int screenWidth = 800;
    const int screenHeight = 600;

    InitWindow(screenWidth, screenHeight, "iloveurmom");
    SetTargetFPS(60);

    while(!WindowShouldClose()){
        BeginDrawing();


            ClearBackground(RAYWHITE);
            DrawCircle(screenWidth/2, screenHeight/2, 100, RED);
            // DrawText("iloveurmom", screenWidth/2, screenHeight/2, 69, LIGHTGRAY);

        EndDrawing();
    }

    CloseWindow();

    return 0;
}
