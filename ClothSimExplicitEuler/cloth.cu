#include "cloth.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>

#include <helper_cuda.h>
#include <helper_math.h>

#include <glm\glm.hpp>

#define NUM_ITERS 8

//kernels

//shader ids
extern GLuint programID, colorID, tex_normal_map, tex_normal_map_id, normal_sign_id;

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
        pos.z = (int)absIndex.y * spring_dim.y * -1;
        break;
    case Cloth::vertical:
    default:
        pos.z = 0.0f;
        pos.x = (int)(absIndex.x - num_particles.x / 2) * spring_dim.x;
        pos.y = (int)(absIndex.y - num_particles.y / 2) * spring_dim.y * -1;
        break;
    }

    positions[absIndex.x + absIndex.y * num_particles.x] = pos;
}

__global__ void k_compute_normals(float4 *positions, float4 *normals, uint2 num_particles)
{
    int2 absIndex;
    absIndex.x = blockIdx.x * blockDim.x + threadIdx.x;
    absIndex.y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int pos;
    pos = absIndex.x + absIndex.y * num_particles.x;

	normals[pos] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
    float4 temp;
	float3 curr_norm;

    if( absIndex.x < ( num_particles.x - 1 ) && absIndex.y < ( num_particles.y - 1 ) )
    {
        unsigned int pos_neigh1, pos_neigh2, pos_neigh3;
        pos_neigh1 = pos + 1;
        pos_neigh2 = pos + num_particles.x; 
        pos_neigh3 = pos + 1 + num_particles.x;

        float3 A,B,C, AB, AC, norm;
        temp = positions[pos];
        A = make_float3( temp.x, temp. y, temp.z );
        temp = positions[pos_neigh2];
        B = make_float3( temp.x, temp. y, temp.z );
        temp = positions[pos_neigh1];
        C = make_float3( temp.x, temp. y, temp.z );

        AB = B - A;
        AC = C - A;

        norm = cross( AB, AC );
        //norm = normalize( norm );

        temp = normals[pos];
        curr_norm = make_float3( temp.x, temp.y, temp.z );
        curr_norm = curr_norm + norm;
        normals[pos] = make_float4( curr_norm.x, curr_norm.y, curr_norm.z, temp.w );
        __syncthreads();
        temp = normals[pos_neigh1];
        curr_norm = make_float3( temp.x, temp.y, temp.z );
        curr_norm = curr_norm + norm;
        normals[pos_neigh1] = make_float4( curr_norm.x, curr_norm.y, curr_norm.z, temp.w );
        __syncthreads();
        temp = normals[pos_neigh2];
        curr_norm = make_float3( temp.x, temp.y, temp.z );
        curr_norm = curr_norm + norm;
        normals[pos_neigh2] = make_float4( curr_norm.x, curr_norm.y, curr_norm.z, temp.w );
        __syncthreads();

        temp = positions[pos_neigh3];
        A = make_float3( temp.x, temp. y, temp.z );
        temp = positions[pos_neigh1];
        B = make_float3( temp.x, temp. y, temp.z );
        temp = positions[pos_neigh2];
        C = make_float3( temp.x, temp. y, temp.z );

        AB = B - A;
        AC = C - A;

        norm = cross( AB, AC );
        norm = normalize( norm );

        temp = normals[pos_neigh3];
        curr_norm = make_float3( temp.x, temp.y, temp.z );
        curr_norm = curr_norm + norm;
        normals[pos_neigh3] = make_float4( curr_norm.x, curr_norm.y, curr_norm.z, temp.w );
        __syncthreads();
        temp = normals[pos_neigh1];
        curr_norm = make_float3( temp.x, temp.y, temp.z );
        curr_norm = curr_norm + norm;
        normals[pos_neigh1] = make_float4( curr_norm.x, curr_norm.y, curr_norm.z, temp.w );
        __syncthreads();
        temp = normals[pos_neigh2];
        curr_norm = make_float3( temp.x, temp.y, temp.z );
        curr_norm = curr_norm + norm;
        normals[pos_neigh2] = make_float4( curr_norm.x, curr_norm.y, curr_norm.z, temp.w );
        __syncthreads();
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

__device__ float3 compute_spring_accelerations( float3 pos_i, float3 v_i, float4 *positions, float4 *velocities, NeighbourData *neighbour_data, unsigned int num_neighbours )
{
    float3 pos_j, x_ij;
    float3 v_j;
    NeighbourData *p = neighbour_data;
    float resting_length, current_length;
    float3 rez = make_float3( 0.0f, 0.0f, 0.0f );
    float4 temp;
    for( unsigned int i  = 0; i < num_neighbours; ++i )
    {
        temp = positions[p->index];
        pos_j = make_float3( temp.x, temp.y, temp.z );
        temp = velocities[p->index];
        v_j = make_float3( temp.x, temp.y, temp.z );

        resting_length = p->rest_length;

        x_ij = pos_j - pos_i;
        current_length = length( x_ij );
        rez = rez + neighbour_data->ks * ( current_length - resting_length ) * normalize( x_ij ) ;
        rez = rez - neighbour_data->kd * ( v_i - v_j );
		
        p++;
    }

    return rez;
}

__device__ float3 compute_wind( float3 normal, unsigned int num_neighbours, float mass )
{
	float3 wind_dir = make_float3( 0.0f, 0.0f, -0.1f );
	return mass * num_neighbours * normal * dot( normal, wind_dir );
}

//euler temporar, doar cu gravitatie
__global__ void euler_integration( float4 *positions, float4 *velocities, float4 *normals, uint2 *neighbours, NeighbourData *neighbourhood, uint2 num_particles, float dt, float damp, float mass )
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
    float3 rez;
    float3 a1;
    float3 a2;
    float3 p1;
    float3 p2;
    float3 v1;
    float3 v2;
    float inv_mass;
	float3 N;

	temp = normals[absId];
	N = make_float3( temp.x, temp.y, temp.z );
    temp = positions[absId];
    p1 = make_float3( temp.x, temp.y, temp.z );
    inv_mass = temp.w;
    temp = velocities[absId];
    v1 = make_float3( temp.x, temp.y, temp.z );

	a1 = make_float3( 0.0f, -0.981f * mass, 0.0f );
	a2 = make_float3( 0.0f, -0.981f * mass, 0.0f );

    uint2 neighbour_data_pointer = neighbours[absId];

	a1 = a1 + v1 * damp + compute_spring_accelerations( p1, v1, positions, velocities, neighbourhood + neighbour_data_pointer.x, neighbour_data_pointer.y ) + compute_wind( N, neighbour_data_pointer.y, mass ); 
    p2 = p1 + dt * v1;
    v2 = v1 + dt * a1 * inv_mass;

    __syncthreads();

    positions[absId] = make_float4( p2.x, p2.y, p2.z, inv_mass );
    velocities[absId] = make_float4( v2.x, v2.y, v2.z , 0.0f );

    __syncthreads();

	a2 = a2 + v2 * damp + compute_spring_accelerations( p2, v2, positions, velocities, neighbourhood + neighbour_data_pointer.x, neighbour_data_pointer.y ) + compute_wind( N, neighbour_data_pointer.y, mass ); 
    rez = p1 + ( v1 + v2 ) * dt * 0.5f;
    positions[absId] = make_float4( rez.x, rez.y, rez.z, inv_mass );
    rez = v1 + ( a1 + a2 ) * dt * 0.5f * inv_mass;
    velocities[absId] = make_float4( rez.x, rez.y, rez.z, 0.0f );
}

//-----

Cloth::Cloth( uint2 dim, uint2 num, float ks_in, float kd_in, float dt_in, float damp_in, float mass_in ) : dimensions( dim ), num_particles( num ), ks( ks_in ), kd( kd_in ), dt( dt_in ), damp( damp_in ), cloth_mass( mass_in ), start_pos( horizontal ), animate( false ), wireframe( true ), fixed_pos( upper_corners )
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
	checkCudaErrors( cudaMemset( d_velocities, 0, sizeof( float4 ) * particle_count ) );
	//shift_x( 1.10f );
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
			uv.push_back( u * x_step * 2 );
			uv.push_back( v * y_step * 2 );
		}
	}

    block = dim3( 16, 16, 1 );
    grid = dim3( num_particles.x / block.x + ( num_particles.x % block.x > 0 ),
                 num_particles.y / block.y + ( num_particles.y % block.y > 0 ),
                 1 );

    checkCudaErrors( cudaMalloc( (void**)&d_velocities, sizeof( float4 ) * particle_count ) );
    checkCudaErrors( cudaMemset( d_velocities, 0, sizeof( float4 ) * particle_count ) );
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

		color = glm::vec3( 0.8f, 0.0f, 0.0f );    

		glUniform3fv( colorID, 1, &color[0] );

		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, element_color1 );
		glDrawElements( GL_TRIANGLES, index_color1.size(), GL_UNSIGNED_SHORT, 0 );
    
        color = glm::vec3( 0.0f, 0.0f, 0.8f );

		glUniform3fv( colorID, 1, &color[0] );

		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, element_color2 );
		glDrawElements( GL_TRIANGLES, index_color2.size(), GL_UNSIGNED_SHORT, 0 );

		glCullFace( GL_FRONT );

		glUniform1i( normal_sign_id, -1 );

		color = glm::vec3( 0.8f, 0.0f, 0.0f );    

		glUniform3fv( colorID, 1, &color[0] );

		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, element_color1 );
		glDrawElements( GL_TRIANGLES, index_color1.size(), GL_UNSIGNED_SHORT, 0 );
    
        color = glm::vec3( 0.0f, 0.0f, 0.8f );

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
	checkCudaErrors( cudaFree( d_velocities ) );
    deleteVBOs();
}

void Cloth::timestep()
{
    float4 *d_positions, *d_normals;
    size_t num_bytes;
    cudaError_t err;

    glBindBuffer( GL_ARRAY_BUFFER, vbo_positions );

    checkCudaErrors( cudaGraphicsMapResources( 1, &vbo_res_positions ) );
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**) &d_positions, &num_bytes, vbo_res_positions ) );
    checkCudaErrors( cudaGraphicsMapResources( 1, &vbo_res_normals ) );
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**) &d_normals, &num_bytes, vbo_res_normals ) );

    for( int i = 0; i < NUM_ITERS; ++i )
    {
        euler_integration<<<grid,block>>>( d_positions, d_velocities, d_normals, d_neighbours, d_neighbourhood, num_particles, dt, damp, particle_mass );

        err = cudaGetLastError();
        if( err )
        {
            std::cout<<"error at integration kernel: "<<cudaGetErrorString( err )<<std::endl;
        }
    }

    k_compute_normals<<<grid,block>>>( d_positions, d_normals, num_particles );
	k_normalize_everything<<<grid,block>>>( d_normals, num_particles );
    err = cudaGetLastError();
    if( err )
    {
      std::cout<<"error at normals kernel: "<<cudaGetErrorString( err )<<std::endl;
    }

    checkCudaErrors( cudaGraphicsUnmapResources( 1, &vbo_res_positions ) );
    checkCudaErrors( cudaGraphicsUnmapResources( 1, &vbo_res_normals ) );
}

void Cloth::init_particles()
{
    float2 spring_dim = make_float2( dimensions.x / (float) num_particles.x, dimensions.y / (float) num_particles.y );
    float4 *d_positions, *d_normals, *h_positions = NULL, *h_normals = NULL, *p;
    size_t num_bytes;

    checkCudaErrors( cudaGraphicsMapResources( 1, &vbo_res_positions ) );
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**) &d_positions, &num_bytes, vbo_res_positions ) );
    std::cout<<"num_bytes: "<<num_bytes<<std::endl;
    checkCudaErrors( cudaGraphicsMapResources( 1, &vbo_res_normals ) );
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**) &d_normals, &num_bytes, vbo_res_normals ) );
    std::cout<<"num_bytes: "<<num_bytes<<std::endl;

	k_cloth_init<<<grid, block>>>( d_positions, spring_dim, num_particles, start_pos, particle_mass, fixed_pos );
    cudaError_t err = cudaGetLastError();
    if( err )
    {
        std::cout<<"error at cloth init kernel: "<<cudaGetErrorString( err )<<std::endl;
    }

    k_compute_normals<<<grid, block>>>(d_positions, d_normals, num_particles);
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
                std::cout<<"("<<p->x<<","<<p->y<<","<<p->z<<") ";
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

    h_neighbours = (uint2*)malloc( particle_count * sizeof( uint2 ) );
    h_positions = (float4*)malloc( num_bytes );
    std::cout<<"num bytes: "<<num_bytes<<std::endl;
    checkCudaErrors( cudaMemcpy( h_positions, d_positions, num_bytes, cudaMemcpyDeviceToHost ) );

    int count = 0;
    for( int y = 0; y < num_particles.y; ++y )
    {
        for( int x = 0; x < num_particles.x; ++x, ++count )
        {
            unsigned int index = x + y * num_particles.x;

            h_neighbours[count].x = h_neighbourhood.size();
            h_neighbours[count].y = 0;
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
                    h_neighbours[count].y++;
                }
            }
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
                h_neighbours[count].y++;
            }
            if( x < num_particles.x - 2 )
            {
                neighbour_data.index = index + 2;
                temp_f4 = h_positions[ neighbour_data.index ];
                temp2_f3 = make_float3( temp_f4.x, temp_f4.y, temp_f4.z );
                neighbour_data.rest_length = length( temp1_f3 - temp2_f3 );
                h_neighbourhood.push_back( neighbour_data );
                h_neighbours[count].y++;
            }
            if( y >= 2)
            {
                neighbour_data.index = index - 2 * num_particles.x;
                temp_f4 = h_positions[ neighbour_data.index ];
                temp2_f3 = make_float3( temp_f4.x, temp_f4.y, temp_f4.z );
                neighbour_data.rest_length = length( temp1_f3 - temp2_f3 );
                h_neighbourhood.push_back( neighbour_data );
                h_neighbours[count].y++;
            }
            if( y < num_particles.y - 2)
            {
                neighbour_data.index = index + 2 * num_particles.x;
                temp_f4 = h_positions[ neighbour_data.index ];
                temp2_f3 = make_float3( temp_f4.x, temp_f4.y, temp_f4.z );
                neighbour_data.rest_length = length( temp1_f3 - temp2_f3 );
                h_neighbourhood.push_back( neighbour_data );
                h_neighbours[count].y++;
            }
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
            std::cout<<"["<<h_neighbours[count].x<<","<<h_neighbours[count].y<<"] ";
            count++;
        }
        std::cout<<std::endl;
    }*/
	checkCudaErrors( cudaMemcpy( d_positions, h_positions, num_bytes, cudaMemcpyHostToDevice ) );
    free( h_positions );

    checkCudaErrors( cudaGraphicsUnmapResources( 1, &vbo_res_positions ) );

    checkCudaErrors( cudaMalloc( (void**)&d_neighbours, sizeof( uint2 ) * particle_count ) );
    checkCudaErrors( cudaMemcpy( d_neighbours, h_neighbours, sizeof( uint2 ) * particle_count, cudaMemcpyHostToDevice ) );

    checkCudaErrors( cudaMalloc( (void**)&d_neighbourhood, sizeof( NeighbourData ) * h_neighbourhood.size() ) );
    checkCudaErrors( cudaMemcpy( d_neighbourhood, &h_neighbourhood[0], sizeof( NeighbourData ) * h_neighbourhood.size(), cudaMemcpyHostToDevice ) );

    free( h_neighbours );
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