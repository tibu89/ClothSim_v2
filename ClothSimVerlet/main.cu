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

//glm
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>

#include "cloth.h"

//time
double last_time = 0.0f, current_time;
int num_frames = 0;

//misc
unsigned int w_width = 512 + 512/2;
unsigned int w_height = 512 + 512/2;

float fov = 60.0f, near_plane = 0.1f, far_plane = 100.0f, ratio;

// mouse
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0f, rotate_y = 3.14159f;
float height = 0.0f, radius = 30.0f;

// forward declarations
bool initGL( int *argc, char **argv );
void display();
void keyboard( unsigned char key, int x, int y );
void mouse( int button, int state, int x, int y );
void motion( int x, int y );
void reshape( int w, int h );

void update_camera_position();

glm::mat4 P, V, M;
glm::vec3 camera_pos;
glm::vec3 focus_pos;
glm::vec3 up;
glm::vec3 right;

glm::vec4 light_pos( 20.0f, 20.0f, 20.0f, 1.0f );

//
Cloth *cloth;

//shader ids
GLuint programID;
GLuint MVPmatrixID, model_matrixID, view_matrixID;
GLuint colorID;
GLuint light_posID;
GLuint normal_sign_id;

GLuint tex_normal_map, tex_normal_map_id;

//ball
float ball_pos[] = {0.0, -7.0, 2.0};
bool ball_exists = false;

enum Mouse_mode
{
	camera_mode,
	select_mode
};

Mouse_mode mouse_mode = Mouse_mode::camera_mode;

float3 ray;

void ray_picker(int x, int y)
{
	glm::vec3 n,f;
	n = glm::unProject( glm::vec3( x, w_height - y - 1, 0.0f ), V * M, P, glm::vec4( 0, 0, w_width, w_height ) );
	f = glm::unProject( glm::vec3( x, w_height - y - 1, 1.0f ), V * M, P, glm::vec4( 0, 0, w_width, w_height ) );
	
	ray = make_float3( f.x - n.x, f.y - n.y, f.z - n.z );
	cloth->pick( make_float3( camera_pos.x, camera_pos.y, camera_pos.z ), ray );
}

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
    glutCreateWindow( "ClothSimVerlet" );
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
	glFrontFace( GL_CW );
    glPolygonMode( GL_FRONT, GL_LINE );

    // viewport
    glViewport(0, 0, w_width, w_height);

    // projection
	ratio = (float) w_width / (float) w_height;
	P = glm::perspective( fov, ratio, near_plane, far_plane );
	update_camera_position();

	M = glm::mat4x4( 1.0 );

	tex_normal_map = SOIL_load_OGL_texture(
		"..\\common\\data\\cloth_normal_map2.jpg",
		SOIL_LOAD_AUTO,
		SOIL_CREATE_NEW_ID,
		SOIL_FLAG_TEXTURE_REPEATS | SOIL_FLAG_MIPMAPS | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT | SOIL_FLAG_INVERT_Y
	);

	if( tex_normal_map == 0 )
	{
		printf( "SOIL load error: '%s'\n", SOIL_last_result() );
	}

	//glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, fLargest );

    SDK_CHECK_ERROR_GL();

    return true;
}

GLuint LoadShaders(const char * vertex_file_path,const char * fragment_file_path){
 
    // Create the shaders
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
 
    // Read the Vertex Shader code from the file
    std::string VertexShaderCode;
    std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
    if(VertexShaderStream.is_open())
    {
        std::string Line = "";
        while(getline(VertexShaderStream, Line))
            VertexShaderCode += "\n" + Line;
        VertexShaderStream.close();
    }
 
    // Read the Fragment Shader code from the file
    std::string FragmentShaderCode;
    std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
    if(FragmentShaderStream.is_open()){
        std::string Line = "";
        while(getline(FragmentShaderStream, Line))
            FragmentShaderCode += "\n" + Line;
        FragmentShaderStream.close();
    }
 
    GLint Result = GL_FALSE;
    int InfoLogLength;
 
    // Compile Vertex Shader
    printf("Compiling shader : %s\n", vertex_file_path);
    char const * VertexSourcePointer = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
    glCompileShader(VertexShaderID);
 
    // Check Vertex Shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    std::vector<char> VertexShaderErrorMessage(InfoLogLength);
    glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
    fprintf(stdout, "%s\n", &VertexShaderErrorMessage[0]);
 
    // Compile Fragment Shader
    printf("Compiling shader : %s\n", fragment_file_path);
    char const * FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
    glCompileShader(FragmentShaderID);
 
    // Check Fragment Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    std::vector<char> FragmentShaderErrorMessage(InfoLogLength);
    glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
    fprintf(stdout, "%s\n", &FragmentShaderErrorMessage[0]);
 
    // Link the program
    fprintf(stdout, "Linking program\n");
    GLuint ProgramID = glCreateProgram();
    glAttachShader(ProgramID, VertexShaderID);
    glAttachShader(ProgramID, FragmentShaderID);
    glLinkProgram(ProgramID);
 
    // Check the program
    glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
    glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    std::vector<char> ProgramErrorMessage( max(InfoLogLength, int(1)) );
    glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
    fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);
 
    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);
 
    return ProgramID;
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

	programID = LoadShaders( "../common/shaders/cloth_vertex.vs", "../common/shaders/cloth_fragment.fs" );
	MVPmatrixID = glGetUniformLocation( programID, "MVP" );
	model_matrixID = glGetUniformLocation( programID, "M" );
	view_matrixID = glGetUniformLocation( programID, "V" );
	colorID = glGetUniformLocation( programID, "my_color" );
	light_posID = glGetUniformLocation( programID, "light_pos_worldspace" );
	tex_normal_map_id = glGetUniformLocation( programID, "normal_map_sampler" );
	normal_sign_id = glGetUniformLocation( programID, "normal_sign" );

	cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );

	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 8, 8 ), 250.25f, 0.25f, 0.02f, -0.0125f, 1024.0f );
	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 16, 16 ), 250.25f, 0.25f, 0.01f, -0.0125f, 1024.0f );
	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 32, 32 ), 250.25f, 0.25f, 0.01f, -0.0125f, 1024.0f );
	cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 64, 64 ), 250.25f, 0.125f, 0.00015f, -0.0125f, 1024.0f );
	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 64, 64 ), 50.25f, 0.25f, 0.01f, -0.0125f, 1024.0f );
	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 128, 128 ), 200.25f, 0.25f, 0.008f, -0.0125f, 1024.0f * 3 );
	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 128, 128 ), 200.25f, 0.40f, 0.008f, -0.0125f, 1024.0f );
	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 256, 256 ), 150.25f, 0.50f, 0.005f, -0.0125f, 1024.0f * 4 );
	//cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 256, 256 ), 50.25f, 0.50f, 0.005f, -0.0125f, 1024.0f * 4 );
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
		cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 64, 64 ), 250.25f, 0.25f, 0.015f, -0.0125f, 1024.0f );
		show_vram_usage();
		break;
	case '4':
		delete cloth;
		cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 128, 128 ), 200.25f, 0.40f, 0.008f, -0.0f, 1024.0f );
		show_vram_usage();
		break;
	case '5':
		delete cloth;
		cloth = new Cloth( make_uint2( 15, 15 ), make_uint2( 256, 256 ), 150.25f, 0.50f, 0.008f, -0.0125f, 1024.0f * 4 );
		show_vram_usage();
		break;
	case 'f':
	case 'F':
		cloth->toggle_fixed_particles();
		cloth->reset( false );
		break;
	case 'd':
	case 'D':
		cloth->toggle_wind();
		break;
    }
}

void mouse(int button, int state, int x, int y)
{
	std::cout<<x<<" "<<y<<std::endl;
	if (state == GLUT_DOWN)
    {
		if( button == GLUT_LEFT_BUTTON )
		{
			ray_picker( x, y );
		}
        if( button == 3 )
        {
            radius -= 0.5;

			if( radius < 1 ) radius = 1.0f;

			update_camera_position();
        }
        else if ( button == 4 )
        {
            radius += 0.5;

			if( radius < 1 ) radius = 1.0f;

			update_camera_position();
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

	switch( mouse_mode )
	{
	case Mouse_mode::camera_mode:
		if (mouse_buttons & 1)
		{
			rotate_x -= dy * 0.01f;
			rotate_y -= dx * 0.01f;

			update_camera_position();
		}
		else if (mouse_buttons & 4)
		{
			focus_pos.y += dy * 0.05f;

			update_camera_position();
		}
		else if (mouse_buttons & 2)
		{
			radius += dy * 0.05f;

			if( radius < 1 ) radius = 1.0f;

			update_camera_position();
		}
		break;
	case Mouse_mode::select_mode:
		break;
	}

    mouse_old_x = x;
    mouse_old_y = y;
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram( programID );

    // set view matrix
    V = glm::lookAt( camera_pos, focus_pos, up );

	glm::mat4 MVP = P * V * M;
	glUniformMatrix4fv( MVPmatrixID, 1, GL_FALSE, &MVP[0][0] );
	glUniformMatrix4fv( model_matrixID, 1, GL_FALSE, &M[0][0] );
	glUniformMatrix4fv( view_matrixID, 1, GL_FALSE, &V[0][0] );
	glUniform4fv( light_posID, 1, &light_pos[0] ); 

    cloth->draw();
	//current_time = glutGet( GLUT_ELAPSED_TIME );
	//light_pos[0] = 20.0f * sin( current_time / 350 );
	glUseProgram( 0 );

	num_frames++;
	if( current_time - last_time > 1000.0f )
	{
		//printf("%f\n", (current_time - last_time) / (double)num_frames );
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
	ratio = (float) w_width / (float) w_height;
	glViewport( 0, 0, (GLsizei)w_width, (GLsizei)w_height );
	P = glm::perspective( fov, ratio, near_plane, far_plane );
}

void update_camera_position()
{
	glm::vec3 direction( cos( rotate_x ) * sin( rotate_y ),
						 sin( rotate_x ),
						 cos( rotate_x ) * cos( rotate_y ) );

	camera_pos = focus_pos - radius * direction;
	right = glm::vec3(	sin( rotate_y - 3.14159f / 2.0f ),
						0,
						cos( rotate_y - 3.14159f / 2.0f ) );
	up = glm::cross( right, direction );
}