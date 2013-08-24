#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WINDOW_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

//openGL includes
#include <GL\glew.h>
#include <GL\freeglut.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <vector_types.h>
#include <vector_functions.h>

//Simple OpenGL Image Library
#include <soil.h>

#include "cloth.h"

//time
double last_time = 0.0f, current_time;
int num_frames = 0;

//misc
unsigned int w_width = 512;
unsigned int w_height = 512;
float light_pos[] = {1.0f, 0.0f, 1.0f, 1.0f};

// mouse
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0f, rotate_y = 0.0f;
float translate_y = 0.0f, translate_z = -30.0f;

// forward declarations
bool initGL( int *argc, char **argv );
void display();
void keyboard( unsigned char key, int x, int y );
void mouse( int button, int state, int x, int y );
void motion( int x, int y );
void reshape( int w, int h );

//
Cloth *cloth;
GLuint tex_2D;

bool checkHW(char *name, const char *gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

int findGraphicsGPU(char *name)
{
    int nGraphicsGPU = 0;
    int deviceCount = 0;
    bool bFoundGraphics = false;
    char firstGraphicsName[256], temp[256];

    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("> There are no device(s) supporting CUDA\n");
        return false;
    }
    else
    {
        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
        printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

        if (bGraphics)
        {
            if (!bFoundGraphics)
            {
                strcpy(firstGraphicsName, temp);
            }

            nGraphicsGPU++;
        }
    }

    if (nGraphicsGPU)
    {
        strcpy(name, firstGraphicsName);
    }
    else
    {
        strcpy(name, "this hardware");
    }

    return nGraphicsGPU;
}

//init openGL
bool initGL(int *argc, char **argv)
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize( w_width, w_height );
    glutCreateWindow( "ClothSimExplicitEuler" );
    glutDisplayFunc( display );
    glutKeyboardFunc( keyboard );
    glutMotionFunc( motion );
	glutMouseFunc( mouse );
	glutReshapeFunc( reshape );
    //glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    glewInit();

    if (! glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }



    // default initialization
    glClearColor(0.52f, 0.80f, 0.97f, 1.0f);
    glEnable( GL_DEPTH_TEST );
	//glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glEnable( GL_COLOR_MATERIAL );
	//glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	//glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	//glEnable( GL_NORMALIZE );
	glEnable( GL_LIGHTING );
	glEnable( GL_LIGHT0 );
	glLightfv( GL_LIGHT0, GL_POSITION, light_pos );
	glDisable( GL_LIGHTING );
	glDisable( GL_LIGHT0 );
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

    // viewport
    glViewport(0, 0, w_width, w_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)w_width / (GLfloat) w_height, 0.1, 1000.0);

    SDK_CHECK_ERROR_GL();

    return true;
}

void show_vram_usage()
{
	size_t free;
    size_t total;
    checkCudaErrors( cudaMemGetInfo( &free, &total) );  

    std::cout << "free memory: " << free / 1024 / 1024 << "mb, total memory: " << total / 1024 / 1024 << "mb" << std::endl;
}

int main( int argc, char **argv )
{
	if( !initGL( &argc, argv ) )
	{
		exit( 1 );
	}

	cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );

	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 8, 8 ), 250.25f, 0.25f, 0.01f, -0.0125f, 1024.0f );
	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 16, 16 ), 250.25f, 0.25f, 0.01f, -0.0125f, 1024.0f );
	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 32, 32 ), 250.25f, 0.25f, 0.01f, -0.0125f, 1024.0f );
	cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 64, 64 ), 250.25f, 0.25f, 0.008f, -0.0125f, 1024.0f );
	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 64, 64 ), 250.25f, 0.25f, 0.01f, -0.0125f, 1024.0f );
	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 128, 128 ), 200.25f, 0.25f, 0.008f, -0.0125f, 1024.0f * 3 );
	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 128, 128 ), 200.25f, 0.40f, 0.008f, -0.0125f, 1024.0f );
	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 256, 256 ), 150.25f, 0.50f, 0.005f, -0.0125f, 1024.0f * 4 );
	show_vram_usage();
	
	last_time = (double)glutGet( GLUT_ELAPSED_TIME );
	glutMainLoop();

	cudaDeviceReset();
	exit(0);
}

//input functions
void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 'w':
    case 'W':
		cloth->toggle_wireframe();
        break;
    case 'p':
    case 'P':
        cloth->toggle_animation();
        break;
    // reset( true ) schimba si pozitia initiala a panzei
    case 'r':
        cloth->reset( false );
        break;
    case 'R':
        cloth->reset( true );
        break;
    case (27) :
        exit(EXIT_SUCCESS);
        break;
	case '1':
		delete cloth;
		cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 16, 16 ), 250.25f, 0.25f, 0.01f, -0.0125f, 1024.0f );
		show_vram_usage();
		break;
	case '2':
		delete cloth;
		cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 32, 32 ), 250.25f, 0.25f, 0.01f, -0.0125f, 1024.0f );
		show_vram_usage();
		break;
	case '3':
		delete cloth;
		cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 64, 64 ), 250.25f, 0.25f, 0.008f, -0.0125f, 1024.0f );
		show_vram_usage();
		break;
	case '4':
		delete cloth;
		cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 128, 128 ), 200.25f, 0.40f, 0.008f, -0.0125f, 1024.0f );
		show_vram_usage();
		break;
	case '5':
		delete cloth;
		cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 256, 256 ), 150.25f, 0.50f, 0.005f, -0.0125f, 1024.0f * 4 );
		show_vram_usage();
		break;
	case 'f':
	case 'F':
		cloth->toggle_fixed_particles();
		cloth->reset( false );
		break;
    }
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        if( button == 3 )
        {
            translate_z += 0.5;
        }
        else if ( button == 4 )
        {
            translate_z -= 0.5;
        }
        else
        {
            mouse_buttons |= 1<<button;
        }
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_y += dy * 0.05f;
    }
    else if (mouse_buttons & 2)
    {
        translate_z -= dy * 0.05f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, translate_y, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    cloth->draw(); 
	current_time = glutGet( GLUT_ELAPSED_TIME );
	num_frames++;
	if( current_time - last_time > 1000.0f )
	{
		printf("%f\n", (current_time - last_time) / (double)num_frames );
		num_frames = 0;
		last_time = current_time;
	}

    glutSwapBuffers();
	glutPostRedisplay();
}

void reshape(int width, int height)
{
	w_width = width;
	w_height = height;
	glViewport( 0, 0, (GLsizei)w_width, (GLsizei)w_height );
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)w_width / (GLfloat) w_height, 0.1, 1000.0);
}