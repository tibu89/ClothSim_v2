#include "cloth.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>

#include <helper_cuda.h>
#include <helper_math.h>

#include <glm\glm.hpp>

#define NUM_ITERS 8
#define MAX_DEFORMATION 0.25f

//kernels

//shader ids
extern GLuint programID, colorID, tex_normal_map, tex_normal_map_id, normal_sign_id;

__device__ bool provot_modif;

__device__ void apply_provot_dynamic_inverse( unsigned int absId, float3 pos, float inv_mass, uint2 num_particles, float4 *positions, NeighbourData *neighbour_data, unsigned int num_neighbours )
{
	if( inv_mass == 0.0f )
		return;

	int2 neigh_index;
	unsigned int neigh_absId;
	float move_ratio;
	float4 temp;
	float3 pos_neigh;
	float neigh_inv_mass;
	float diff;
	float3 p1p2;
	NeighbourData neigh_data;
	float max_def = 1 + MAX_DEFORMATION, min_def = 1 - MAX_DEFORMATION;

	for( int i = 0; i < num_neighbours; i++ )
	{
		neigh_data = neighbour_data[i];
		temp = positions[ neigh_data.index ];
		pos_neigh = make_float3( temp.x, temp.y, temp.z );
		neigh_inv_mass = temp.w;
		move_ratio = inv_mass / ( inv_mass + neigh_inv_mass );
		
		p1p2 = pos_neigh - pos;
		diff = p1p2.x * p1p2.x + p1p2.y * p1p2.y + p1p2.z * p1p2.z;

		if( diff <= ( neigh_data.rest_length * neigh_data.rest_length * max_def * max_def ) &&
			diff >= ( neigh_data.rest_length * neigh_data.rest_length * min_def * min_def ) )
		{
			continue;
		}

		diff = sqrt( diff );
		diff -= neigh_data.rest_length;
		diff *= move_ratio;

		pos += diff * normalize( p1p2 );
		positions[absId] = make_float4( pos, inv_mass );
	}
	
}

__global__ void k_cloth_init( float4 *positions, float2 spring_dim, uint2 num_particles, Cloth::starting_position start_pos, float mass, Cloth::fixed_particles fixed)
{
    int2 absIndex;
    absIndex.x = blockIdx.x * blockDim.x + threadIdx.x;
    absIndex.y = blockIdx.y * blockDim.y + threadIdx.y;
	bool is_fixed = false;

    if( absIndex.x >= num_particles.x || absIndex.y >= num_particles.y )
        return;

    float4 pos;
	switch( fixed )
	{
	case Cloth::upper_corners:
		if( ( absIndex.y == 0 ) && ( absIndex.x == 0 || absIndex.x == ( num_particles.x - 1 ) ) )
		{
			is_fixed = true;
		}
		break;
	case Cloth::upper_edge:
		if( absIndex.y == 0 )
		{
			is_fixed = true;
		}
		break;
	}

    if( is_fixed )
    {
        pos.w = 0;
    }	
    else
    {
        pos.w = 1.0f / mass;
    }

    switch( start_pos )
    {
    case Cloth::horizontal:
        pos.y = 0.0f;
        pos.x = (int)(absIndex.x - num_particles.x / 2) * spring_dim.x;
        pos.z = (int)absIndex.y * spring_dim.y;
        break;
    case Cloth::vertical:
    default:
        pos.z = 0.0f;
        pos.x = (int)(absIndex.x - num_particles.x / 2) * spring_dim.x;
        pos.y = (int) absIndex.y * spring_dim.y * -1;
        break;
    }

    positions[absIndex.x + absIndex.y * num_particles.x] = pos;
}

__global__ void k_compute_triangle_normals(float4 *positions, TrianglePairData *triangle_normals, uint2 num_particles)
{
    int2 absIndex;
    absIndex.x = blockIdx.x * blockDim.x + threadIdx.x;
    absIndex.y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int pos;
    pos = absIndex.x + absIndex.y * num_particles.x;

    float4 temp;
    if( absIndex.x < ( num_particles.x - 1 ) && absIndex.y < ( num_particles.y - 1 ) )
    {
        unsigned int pos_neigh1, pos_neigh2, pos_neigh3;
        pos_neigh1 = pos + 1;
        pos_neigh2 = pos + num_particles.x; 
        pos_neigh3 = pos + 1 + num_particles.x;

        float3 A,B,C, AB, AC, norm;
        temp = positions[pos];
        A = make_float3( temp.x, temp.y, temp.z );
        temp = positions[pos_neigh2];
        B = make_float3( temp.x, temp.y, temp.z );
        temp = positions[pos_neigh1];
        C = make_float3( temp.x, temp.y, temp.z );

        AB = B - A;
        AC = C - A;

        norm = cross( AB, AC );
		temp.w = length( norm ) / 2;
		norm = normalize( norm );
		temp.x = norm.x; temp.y = norm.y; temp.z = norm.z;
		triangle_normals[pos].N[0] = temp;

        temp = positions[pos_neigh3];
        A = make_float3( temp.x, temp.y, temp.z );
        temp = positions[pos_neigh1];
        B = make_float3( temp.x, temp.y, temp.z );
        temp = positions[pos_neigh2];
        C = make_float3( temp.x, temp.y, temp.z );

        AB = B - A;
        AC = C - A;

        norm = cross( AB, AC );
		temp.w = length( norm ) / 2;
		norm = normalize( norm );
		temp.x = norm.x; temp.y = norm.y; temp.z = norm.z;
		triangle_normals[pos].N[1] = temp;
    }
}

__device__ unsigned int offset_x[] = { 0, 1, 0, 1 };
__device__ unsigned int offset_y[] = { 0, 0, 1, 1 };

__global__ void k_compute_normals_no_overlap( TrianglePairData *triangle_normals, float4 *normals, uint2 num_particles, unsigned int offset_index, int tr )
{
	int2 absIndex, normal_index;
    absIndex.x = blockIdx.x * blockDim.x + threadIdx.x;
    absIndex.y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int off_x = offset_x[offset_index], off_y = offset_y[offset_index];

	normal_index.x = absIndex.x * 2 + off_x;
	normal_index.y = absIndex.y * 2 + off_y;

	float4 temp;

	if( normal_index.x < ( num_particles.x - 1 ) && normal_index.y < ( num_particles.y - 1 ) )
	{
		unsigned int pos = normal_index.x  + normal_index.y * num_particles.x;
		unsigned p1, p2, p3;

		if( tr == 0 )
		{
			p1 = pos;
			p2 = pos + 1;
			p3 = pos + num_particles.x;
		}
		else
		{
			p1 = pos + 1;
			p2 = pos + num_particles.x;
			p3 = pos + num_particles.x + 1;
		}

		temp = triangle_normals[pos].N[tr];
		
		normals[p1] += temp;
		normals[p2] += temp;
		normals[p3] += temp;
	}	
}

__global__ void k_normalize_everything( float4 *normals, uint2 num_particles )
{
	int2 absIndex;
    absIndex.x = blockIdx.x * blockDim.x + threadIdx.x;
    absIndex.y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int pos;
    pos = absIndex.x + absIndex.y * num_particles.x;

	if( absIndex.x >= num_particles.x || absIndex.y >= num_particles.y )
        return;

    float4 temp = normals[pos];
    float3 curr_norm = normalize( make_float3( temp.x, temp.y, temp.z ) );
    
    normals[pos] = make_float4( curr_norm.x, curr_norm.y, curr_norm.z, temp.w );
}

__device__ float3 compute_spring_accelerations( float3 pos_i, float3 pos_old_i, float4 *positions, float4 *positions_old, NeighbourData *neighbour_data, unsigned int num_neighbours, float dt )
{
    float3 pos_j, x_ij;
    float3 v_i, v_j;
    NeighbourData *p = neighbour_data;
    float resting_length, current_length;
    float3 rez = make_float3( 0.0f, 0.0f, 0.0f );
    float4 temp;

	v_i = ( pos_i - pos_old_i ) / dt;
    for( unsigned int i  = 0; i < num_neighbours; ++i )
    {
        temp = positions[p->index];
        pos_j = make_float3( temp.x, temp.y, temp.z );
        temp = positions_old[p->index];
        v_j = ( pos_j - make_float3( temp.x, temp.y, temp.z ) ) / dt;

        resting_length = p->rest_length;

        x_ij = pos_j - pos_i;
        current_length = length( x_ij );
        rez = rez + neighbour_data->ks * ( current_length - resting_length ) * normalize( x_ij ) ;
        rez = rez - neighbour_data->kd * ( v_i - v_j );
		
        p++;
    }

    return rez;
}

__device__ float3 compute_wind( float4 normal )
{
	float3 wind_dir = make_float3( 0.0f, 0.0f, -1.0f );
	float3 N = make_float3( normal.x, normal.y, normal.z );
	float surrounding_area = normal.w;
	return 1.5f * surrounding_area * N * dot( N, wind_dir );
}

__global__ void k_verlet_integration( float4 *positions_out, float4 *positions_current, float4 *positions_old, float4 *normals, NeighbourDataPointer *neighbours, NeighbourData *neighbourhood, uint2 num_particles, float dt, float damp, float mass, unsigned int wind )
{
    uint2 absIndex;
    unsigned int absId;

    absIndex.x = threadIdx.x + blockDim.x * blockIdx.x;
    absIndex.y = threadIdx.y + blockDim.y * blockIdx.y;

    if( absIndex.x >= num_particles.x || absIndex.y >= num_particles.y )
    {
        return;
    }

    absId = absIndex.y * num_particles.x + absIndex.x;

    float4 temp;
	float3 f = make_float3( 0.0f, -0.981f * mass, 0.0f );
    float3 pos, pos_old;
    float3 v;
    float inv_mass;
	float4 N;

	N = normals[absId];
    temp = positions_current[absId];
    pos = make_float3( temp.x, temp.y, temp.z );
    inv_mass = temp.w;
    temp = positions_old[absId];
    pos_old = make_float3( temp.x, temp.y, temp.z );

	v = ( pos - pos_old ) / dt;

	NeighbourDataPointer neighbour_data_pointer = neighbours[absId];

	f = f + v * damp + compute_spring_accelerations( pos, pos_old, positions_current, positions_old, neighbourhood + neighbour_data_pointer.index, neighbour_data_pointer.neighbour_count, dt ) + wind * compute_wind( N ); 
    pos = pos + ( pos - pos_old ) + f * inv_mass * dt * dt;

	positions_out[absId] = make_float4( pos.x, pos.y, pos.z, inv_mass );
	for( int i = 0; i < 8; ++i )
	{
		//apply_provot_dynamic_inverse( absId, inv_mass, num_particles, positions_out, neighbourhood + neighbour_data_pointer.index, neighbour_data_pointer.near_neighbour_count );
	}
/*
	float3 center = make_float3(0.0, -7.0, 2.0);
	float radius = 5;

	if (length(pos - center) < radius)
	{
		// collision
		float3 coll_dir = normalize(pos - center);
		pos = center + coll_dir * radius;
	}

	positions_out[absId] = make_float4( pos.x, pos.y, pos.z, inv_mass );*/
}

//-----

Cloth::Cloth( uint2 dim, uint2 num, float ks_in, float kd_in, float dt_in, float damp_in, float mass_in ) : dimensions( dim ), num_particles( num ), ks( ks_in ), kd( kd_in ), dt( dt_in ), damp( damp_in ), cloth_mass( mass_in ), start_pos( horizontal ), animate( false ), wireframe( true ), fixed_pos( upper_corners ), wind( false )
{
    init_cloth();
    createVBOs();
    init_particles();
    set_neighbours();
	//shift_x( 1.10f );
	std::cout<<"ks "<<ks<<" kd "<<kd<<" dt "<<dt<<" damp "<<damp<<std::endl;
}

void Cloth::reset( bool change_pos )
{
    animate = false;
    if( change_pos )
    {
        start_pos = (starting_position)(start_pos + 1);
        start_pos = (starting_position)(start_pos % num_pos);
    }
	init_particles();
	//shift_x( 2.0f );
}

void Cloth::toggle_fixed_particles()
{
	fixed_pos = ( fixed_particles )( fixed_pos + 1 );
	fixed_pos = ( fixed_particles )( fixed_pos % num_fixed_pos );
}

void Cloth::init_cloth()
{
    particle_count = num_particles.x * num_particles.y;
	particle_mass = cloth_mass / particle_count;

	GLfloat x_step = 1.0f / (GLfloat)num_particles.x, y_step = 1.0f / (GLfloat)num_particles.y;

    for( unsigned int v = 0; v < num_particles.y - 1; ++v )
    {
        for( unsigned int u = 0; u < num_particles.x - 1; ++u )
        {
            std::vector<unsigned short> *vec;
            if( u % 10 < 5 )
            {
                vec = &index_color1;
            }
            else
            {
                vec = &index_color2;
            }
            vec->push_back( u +     v      * num_particles.x );
            vec->push_back( u + 1 + v      * num_particles.x );
            vec->push_back( u +    (v + 1) * num_particles.x );

            vec->push_back( u + 1 +(v + 1) * num_particles.x );
            vec->push_back( u +    (v + 1) * num_particles.x );
            vec->push_back( u + 1 + v      * num_particles.x );
        }
    }

	for( unsigned int v = 0; v < num_particles.y; ++v )
	{
		for( unsigned int u = 0; u < num_particles.x; ++u )
		{
			uv.push_back( u * x_step * 4 );
			uv.push_back( v * y_step * 4 );
		}
	}

    block = dim3( 16, 16, 1 );
    grid = dim3( num_particles.x / block.x + ( num_particles.x % block.x > 0 ),
                 num_particles.y / block.y + ( num_particles.y % block.y > 0 ),
                 1 );

	half_grid = dim3( ( num_particles.x + 1 ) / 2 / block.x + ( ( num_particles.x + 1 ) / 2 % block.x > 0 ),
					( num_particles.y + 1 ) / 2 / block.y + ( ( num_particles.y + 1 ) / 2 % block.y > 0 ),
					1 );

	checkCudaErrors( cudaMalloc( (void**)&d_triangle_normals, sizeof( TrianglePairData ) * ( num_particles.x - 1 ) * ( num_particles.y - 1 ) ) );
	checkCudaErrors( cudaMalloc( (void**)&d_positions_current, sizeof( float4 ) * particle_count ) );
	checkCudaErrors( cudaMalloc( (void**)&d_positions_old, sizeof( float4 ) * particle_count ) );
}

void Cloth::createVBOs()
{
    unsigned int size = num_particles.x * num_particles.y * sizeof( float4 );
    glGenBuffers( 1, &vbo_positions );

    glBindBuffer( GL_ARRAY_BUFFER, vbo_positions );
    glBufferData( GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW );
    checkCudaErrors( cudaGraphicsGLRegisterBuffer( &vbo_res_positions, vbo_positions, cudaGraphicsRegisterFlagsNone ) );

    glGenBuffers( 1, &vbo_normals );

    glBindBuffer( GL_ARRAY_BUFFER, vbo_normals );
    glBufferData( GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW );
    checkCudaErrors( cudaGraphicsGLRegisterBuffer( &vbo_res_normals, vbo_normals, cudaGraphicsRegisterFlagsNone ) );

	glGenBuffers( 1, &vbo_uv );

	glBindBuffer( GL_ARRAY_BUFFER, vbo_uv );
	glBufferData( GL_ARRAY_BUFFER, uv.size() * sizeof( GLfloat ), &uv[0], GL_STATIC_DRAW );

    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    glGenBuffers( 1, &element_color1 );
    glGenBuffers( 1, &element_color2 );

    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, element_color1 );
    glBufferData( GL_ELEMENT_ARRAY_BUFFER, index_color1.size() * sizeof( unsigned short ), &index_color1[0], GL_STATIC_DRAW );

    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, element_color2 );
    glBufferData( GL_ELEMENT_ARRAY_BUFFER, index_color2.size() * sizeof( unsigned short ), &index_color2[0], GL_STATIC_DRAW );

    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
}

void Cloth::deleteVBOs()
{
    checkCudaErrors( cudaGraphicsUnregisterResource( vbo_res_positions ) );
    checkCudaErrors( cudaGraphicsUnregisterResource( vbo_res_normals ) );

    glBindBuffer( GL_ARRAY_BUFFER, vbo_positions );
    glDeleteBuffers( 1, &vbo_positions );

    glBindBuffer( GL_ARRAY_BUFFER, vbo_normals );
    glDeleteBuffers( 1, &vbo_normals );

    glBindBuffer( GL_ARRAY_BUFFER, 0 );
}

void Cloth::draw()
{
	glm::vec3 color( 0.0f, 0.0f, 0.0f );

	glEnableVertexAttribArray( 0 );
    glBindBuffer( GL_ARRAY_BUFFER, vbo_positions );
	glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, sizeof( float4 ) , (void*)0 );

	glEnableVertexAttribArray( 1 );
    glBindBuffer( GL_ARRAY_BUFFER, vbo_normals );
	glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, sizeof( float4 ) , (void*)0 );

	glEnableVertexAttribArray( 2 );
	glBindBuffer( GL_ARRAY_BUFFER, vbo_uv );
	glVertexAttribPointer( 2, 2, GL_FLOAT, GL_FALSE, 0, (void*)0 );

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, tex_normal_map );
	glUniform1i( tex_normal_map_id, 0 );
    
	if( animate ) 
    {
        timestep();
    }

	if( !wireframe )
    {
		glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

		glEnable( GL_CULL_FACE );
		glCullFace( GL_BACK );

		glUniform1i( normal_sign_id, 1 );

		color = glm::vec3( 0.75f, 0.05f, 0.08f );    

		glUniform3fv( colorID, 1, &color[0] );

		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, element_color1 );
		glDrawElements( GL_TRIANGLES, index_color1.size(), GL_UNSIGNED_SHORT, 0 );
    
        color = glm::vec3( 0.09f, 0.15f, 0.42f );

		glUniform3fv( colorID, 1, &color[0] );

		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, element_color2 );
		glDrawElements( GL_TRIANGLES, index_color2.size(), GL_UNSIGNED_SHORT, 0 );

		glCullFace( GL_FRONT );

		glUniform1i( normal_sign_id, -1 );

		color = glm::vec3( 0.75f, 0.05f, 0.08f );    

		glUniform3fv( colorID, 1, &color[0] );

		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, element_color1 );
		glDrawElements( GL_TRIANGLES, index_color1.size(), GL_UNSIGNED_SHORT, 0 );
    
        color = glm::vec3( 0.09f, 0.15f, 0.42f );

		glUniform3fv( colorID, 1, &color[0] );

		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, element_color2 );
		glDrawElements( GL_TRIANGLES, index_color2.size(), GL_UNSIGNED_SHORT, 0 );

		glDisable( GL_CULL_FACE );
	}
	else
	{
		glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

		glUniform3fv( colorID, 1, &color[0] );

		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, element_color1 );
		glDrawElements( GL_TRIANGLES, index_color1.size(), GL_UNSIGNED_SHORT, 0 );
    
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, element_color2 );
		glDrawElements( GL_TRIANGLES, index_color2.size(), GL_UNSIGNED_SHORT, 0 );
	}

    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );

	glDisableVertexAttribArray( 0 );
	glDisableVertexAttribArray( 1 );
	glDisableVertexAttribArray( 2 );
}

Cloth::~Cloth()
{
	checkCudaErrors( cudaFree( d_neighbourhood ) );
	checkCudaErrors( cudaFree( d_neighbours ) );
	checkCudaErrors( cudaFree( d_triangle_normals ) );
    deleteVBOs();
}

void Cloth::timestep()
{
    float4 *d_positions, *d_normals;
    size_t num_bytes;
    cudaError_t err;

    checkCudaErrors( cudaGraphicsMapResources( 1, &vbo_res_positions ) );
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**) &d_positions, &num_bytes, vbo_res_positions ) );
    checkCudaErrors( cudaGraphicsMapResources( 1, &vbo_res_normals ) );
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**) &d_normals, &num_bytes, vbo_res_normals ) );

    k_compute_triangle_normals<<<grid,block>>>( d_positions, d_triangle_normals, num_particles );
	checkCudaErrors( cudaMemset( d_normals, 0, num_bytes ) );
	if( !wireframe || wind )
	{
		for( int tr = 0; tr < 2; tr++ )
		{
			for( int off = 0; off < 4; off++ )
			{
				k_compute_normals_no_overlap<<<half_grid, block>>>( d_triangle_normals, d_normals, num_particles, off, tr );
			}
		}
		k_normalize_everything<<<grid,block>>>( d_normals, num_particles );
	}
    err = cudaGetLastError();
    if( err )
    {
		std::cout<<"error at normals kernel: "<<cudaGetErrorString( err )<<std::endl;
    }

    for( int i = 0; i < NUM_ITERS; ++i )
    {
        k_verlet_integration<<<grid,block>>>( d_positions, d_positions_current, d_positions_old, d_normals, d_neighbours, d_neighbourhood, num_particles, dt, damp, particle_mass, unsigned int( wind ) );

        err = cudaGetLastError();
        if( err )
        {
            std::cout<<"error at integration kernel: "<<cudaGetErrorString( err )<<std::endl;
        }

		d_positions_temp = d_positions_old;
		d_positions_old = d_positions_current;
		d_positions_current = d_positions_temp;
		checkCudaErrors( cudaMemcpy( d_positions_current, d_positions, num_bytes, cudaMemcpyDeviceToDevice ) );
    }

    checkCudaErrors( cudaGraphicsUnmapResources( 1, &vbo_res_positions ) );
    checkCudaErrors( cudaGraphicsUnmapResources( 1, &vbo_res_normals ) );
}

void Cloth::init_particles()
{
    float2 spring_dim = make_float2( dimensions.x / (float) num_particles.x, dimensions.y / (float) num_particles.y );
    float4 *d_positions, *d_normals, *h_positions = NULL, *h_normals = NULL, *p;
    size_t num_bytes;
	TrianglePairData *h_triangle_normals = NULL, *p2;

    checkCudaErrors( cudaGraphicsMapResources( 1, &vbo_res_positions ) );
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**) &d_positions, &num_bytes, vbo_res_positions ) );

	checkCudaErrors( cudaGraphicsMapResources( 1, &vbo_res_normals ) );
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**) &d_normals, &num_bytes, vbo_res_normals ) );
    //std::cout<<"num_bytes: "<<num_bytes<<std::endl;

	k_cloth_init<<<grid, block>>>( d_positions, spring_dim, num_particles, start_pos, particle_mass, fixed_pos );
    cudaError_t err = cudaGetLastError();
    if( err )
    {
        std::cout<<"error at cloth init kernel: "<<cudaGetErrorString( err )<<std::endl;
    }

	checkCudaErrors( cudaMemcpy( d_positions_current, d_positions, num_bytes, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( d_positions_old, d_positions, num_bytes, cudaMemcpyDeviceToDevice ) );

    k_compute_triangle_normals<<<grid, block>>>(d_positions, d_triangle_normals, num_particles);
	checkCudaErrors( cudaMemset( d_normals, 0, num_bytes ) );
	for( int tr = 0; tr < 2; tr++ )
	{
		for( int off = 0; off < 4; off++ )
		{
			k_compute_normals_no_overlap<<<half_grid, block>>>( d_triangle_normals, d_normals, num_particles, off, tr );
		}
	}
	k_normalize_everything<<<grid,block>>>( d_normals, num_particles );
    err = cudaGetLastError();
    if( err )
    {
        std::cout<<"error at compute normals kernel: "<<cudaGetErrorString( err )<<std::endl;
    }

    //h_normals = (float4*)malloc( particle_count * sizeof( float4 ) );
    if( h_normals )
    {
        p = h_normals;
        checkCudaErrors( cudaMemcpy( h_normals, d_normals, particle_count * sizeof( float4 ), cudaMemcpyDeviceToHost ) );
        for( int i = 0; i < num_particles.y; ++i )
        {
            for( int j = 0; j < num_particles.x; ++j )
            {
                std::cout<<"("<<p->x<<","<<p->y<<","<<p->z<<","<<p->w<<") ";
                p++;
            }
            std::cout<<std::endl;
        }

        free( h_normals );
    }
    //h_positions = (float4*)malloc( particle_count * sizeof( float4 ) );
    if( h_positions )
    {
        p = h_positions;
        checkCudaErrors( cudaMemcpy( h_positions, d_positions, particle_count * sizeof( float4 ), cudaMemcpyDeviceToHost ) );
        for( int i = 0; i < num_particles.y; ++i )
        {
            for( int j = 0; j < num_particles.x; ++j )
            {
                std::cout<<"("<<p->x<<","<<p->y<<","<<p->z<<","<<p->w<<") ";
                p++;
            }
            std::cout<<std::endl;

        }

        free( h_positions );
    }

	//h_triangle_normals = (TrianglePairData*)malloc( sizeof( h_triangle_normals ) * (num_particles.x - 1) * (num_particles.y - 1) );
	if( h_triangle_normals )
	{
		p2 = h_triangle_normals;
		checkCudaErrors( cudaMemcpy( h_triangle_normals, d_triangle_normals, sizeof( h_triangle_normals ) * (num_particles.x - 1) * (num_particles.y - 1), cudaMemcpyDeviceToHost ) );
		for( int i = 0; i < (num_particles.y - 1); ++i )
        {
            for( int j = 0; j < (num_particles.x-1); ++j )
            {
				std::cout<<"["<<p2->N[0].x<<","<<p2->N[0].y<<","<<p2->N[0].z<<","<<p2->N[0].w<<"]["<<p2->N[1].x<<","<<p2->N[1].y<<","<<p2->N[1].z<<","<<p2->N[1].w<<"] ";
            }
            std::cout<<std::endl;
        }
	}

    checkCudaErrors( cudaGraphicsUnmapResources( 1, &vbo_res_positions ) );
    checkCudaErrors( cudaGraphicsUnmapResources( 1, &vbo_res_normals ) );

}

void Cloth::set_neighbours()
{
    float4 *h_positions, *d_positions, temp_f4;
    float3 temp1_f3, temp2_f3;
    size_t num_bytes;

    checkCudaErrors( cudaGraphicsMapResources( 1, &vbo_res_positions ) );
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**) &d_positions, &num_bytes, vbo_res_positions ) );

    h_positions = (float4*)malloc( num_bytes );
    std::cout<<"num bytes: "<<num_bytes<<std::endl;
    checkCudaErrors( cudaMemcpy( h_positions, d_positions, num_bytes, cudaMemcpyDeviceToHost ) );

    int count = 0;
    for( int y = 0; y < num_particles.y; ++y )
    {
        for( int x = 0; x < num_particles.x; ++x, ++count )
        {
            unsigned int index = x + y * num_particles.x;
			NeighbourDataPointer new_ndp;

			new_ndp.index = h_neighbourhood.size();
			new_ndp.neighbour_count = 0;
            temp_f4 = h_positions[ index ];
            temp1_f3 = make_float3( temp_f4.x, temp_f4.y, temp_f4.z );
            for( int i = max( x - 1, 0 ); i <= min( x + 1, num_particles.x - 1 ); ++i )
            {
                for( int j = max( y - 1, 0 ); j <= min( y + 1, num_particles.y - 1 ); ++j )
                {
                    if( x == i && y == j) continue;
                    NeighbourData neighbour_data;
                    neighbour_data.index = i + j * num_particles.x;
                    temp_f4 = h_positions[ neighbour_data.index ];
                    temp2_f3 = make_float3( temp_f4.x, temp_f4.y, temp_f4.z );
                    neighbour_data.rest_length = length( temp1_f3 - temp2_f3 );
					neighbour_data.ks = ks;
					neighbour_data.kd = kd;
                    h_neighbourhood.push_back( neighbour_data );
                    //std::cout<<"["<<neighbour_data.index<<","<<neighbour_data.rest_length<<"] ";
                    new_ndp.neighbour_count++;
                }
            }
			new_ndp.near_neighbour_count = new_ndp.neighbour_count;
            NeighbourData neighbour_data;
			neighbour_data.ks = ks;
			neighbour_data.kd = kd;
            if( x >= 2)
            {
                neighbour_data.index = index - 2;
                temp_f4 = h_positions[ neighbour_data.index ];
                temp2_f3 = make_float3( temp_f4.x, temp_f4.y, temp_f4.z );
                neighbour_data.rest_length = length( temp1_f3 - temp2_f3 );
                h_neighbourhood.push_back( neighbour_data );
                new_ndp.neighbour_count++;
            }
            if( x < num_particles.x - 2 )
            {
                neighbour_data.index = index + 2;
                temp_f4 = h_positions[ neighbour_data.index ];
                temp2_f3 = make_float3( temp_f4.x, temp_f4.y, temp_f4.z );
                neighbour_data.rest_length = length( temp1_f3 - temp2_f3 );
                h_neighbourhood.push_back( neighbour_data );
                new_ndp.neighbour_count++;
            }
            if( y >= 2)
            {
                neighbour_data.index = index - 2 * num_particles.x;
                temp_f4 = h_positions[ neighbour_data.index ];
                temp2_f3 = make_float3( temp_f4.x, temp_f4.y, temp_f4.z );
                neighbour_data.rest_length = length( temp1_f3 - temp2_f3 );
                h_neighbourhood.push_back( neighbour_data );
                new_ndp.neighbour_count++;
            }
            if( y < num_particles.y - 2)
            {
                neighbour_data.index = index + 2 * num_particles.x;
                temp_f4 = h_positions[ neighbour_data.index ];
                temp2_f3 = make_float3( temp_f4.x, temp_f4.y, temp_f4.z );
                neighbour_data.rest_length = length( temp1_f3 - temp2_f3 );
                h_neighbourhood.push_back( neighbour_data );
                new_ndp.neighbour_count++;
            }
			h_neighbours.push_back( new_ndp );
            //std::cout<<std::endl;			
        }		
    }
    //std::cout<<std::endl;	
    //test
    /*
    count = 0;
    for( int y = 0; y < num_particles.y; ++y )
    {
        for( int x = 0; x < num_particles.x; ++x )
        {
			std::cout<<"["<<h_neighbours[count].index<<","<<h_neighbours[count].neighbour_count<<","<<h_neighbours[count].near_neighbour_count<<"] ";
            count++;
        }
        std::cout<<std::endl;
    }*/
	checkCudaErrors( cudaMemcpy( d_positions, h_positions, num_bytes, cudaMemcpyHostToDevice ) );
    free( h_positions );

    checkCudaErrors( cudaGraphicsUnmapResources( 1, &vbo_res_positions ) );

	checkCudaErrors( cudaMalloc( (void**)&d_neighbours, sizeof( NeighbourDataPointer ) * particle_count ) );
    checkCudaErrors( cudaMemcpy( d_neighbours, &h_neighbours[0], sizeof( NeighbourDataPointer ) * particle_count, cudaMemcpyHostToDevice ) );

    checkCudaErrors( cudaMalloc( (void**)&d_neighbourhood, sizeof( NeighbourData ) * h_neighbourhood.size() ) );
    checkCudaErrors( cudaMemcpy( d_neighbourhood, &h_neighbourhood[0], sizeof( NeighbourData ) * h_neighbourhood.size(), cudaMemcpyHostToDevice ) );

	h_neighbours.clear();
    h_neighbourhood.clear();
}

void Cloth::shift_x( float scale )
{
	float4 *h_positions, *d_positions;
    size_t num_bytes;

    checkCudaErrors( cudaGraphicsMapResources( 1, &vbo_res_positions ) );
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**) &d_positions, &num_bytes, vbo_res_positions ) );

    h_positions = (float4*)malloc( num_bytes );
    std::cout<<"num bytes: "<<num_bytes<<std::endl;
    checkCudaErrors( cudaMemcpy( h_positions, d_positions, num_bytes, cudaMemcpyDeviceToHost ) );

	for( int y = 0; y < num_particles.y; ++y )
	{
		for( int x = 0; x < num_particles.x; ++x )
		{
			unsigned int index = x + y * num_particles.x;
			h_positions[index].x *= scale;
		}
	}

	checkCudaErrors( cudaMemcpy( d_positions, h_positions, num_bytes, cudaMemcpyHostToDevice ) );
	free( h_positions );
    checkCudaErrors( cudaGraphicsUnmapResources( 1, &vbo_res_positions ) );
}