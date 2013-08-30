#pragma once

#include <stdlib.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>

struct NeighbourData
{
    unsigned int index;
    float rest_length;
	float ks, kd;
};

struct TrianglePairData
{
	float4 N[2];
};

class Cloth
{
public:
    enum starting_position
    {
        horizontal,
        vertical,
        num_pos
    };

	enum fixed_particles
	{
		upper_corners,
		upper_edge,
		num_fixed_pos
	};

    void reset( bool change_pos );
	void toggle_fixed_particles();
    void toggle_animation(){ animate = !animate; }
	void toggle_wireframe()
	{
		wireframe = !wireframe;
	}
	void toggle_wind()
	{
		wind = !wind;
	}

private:
    starting_position start_pos;
	fixed_particles fixed_pos;
    bool animate, wireframe, wind;
    float ks, kd, dt, damp;
	float cloth_mass, particle_mass;
    float4 *d_velocities;
    unsigned char *h_movable_particles;
    unsigned char *d_movable_particles;

    uint2 *h_neighbours, *d_neighbours;
    std::vector<NeighbourData> h_neighbourhood;
    NeighbourData *d_neighbourhood;
	TrianglePairData *d_triangle_normals;
	float4 *d_wind_str;

    std::vector<unsigned short> index_color1, index_color2;
	std::vector<GLfloat> uv;

    uint2 dimensions;
    uint2 num_particles;

    unsigned int particle_count;

    dim3 block;
    dim3 grid;

	dim3 half_grid;

    unsigned int num_triangles_half;

    float cloth_weight;

    GLuint vbo_positions, vbo_normals, vbo_uv;

    GLuint element_color1;
    GLuint element_color2;

    cudaGraphicsResource *vbo_res_positions, *vbo_res_normals;

    void init_particles();
    void init_cloth();
    void createVBOs();
    void deleteVBOs();

    void set_neighbours();
	void shift_x( float scale );

    void timestep();


public:

    Cloth( uint2 dim, uint2 num, float ks, float kd, float dt, float damp, float mass );
    ~Cloth();

    void draw();

	void set_ks( float ks_in ){ ks = ks_in; }
	void set_kd( float kd_in ){ kd = kd_in; }
	void set_dt( float dt_in ){ dt = dt_in; }
	void set_damp( float damp_in ){ damp = damp_in; }
	void set_mass( float mass_in ){ cloth_mass = mass_in; }
};