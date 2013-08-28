#version 330 core

layout( location = 0 ) in vec3 vertex_pos;
layout( location = 1 ) in vec3 vertex_normal;
layout( location = 2 ) in vec2 vertex_uv;

uniform mat4 MVP;
uniform mat4 V;
uniform mat4 M;
uniform vec3 my_color;
uniform vec4 light_pos_worldspace;

out vec2 uv;

out vec3 eye_dir_cameraspace;
out vec3 light_dir_cameraspace;
out vec3 normal_cameraspace;

void main()
{
	gl_Position = MVP * vec4( vertex_pos, 1 );
	
	vec3 vertex_pos_cameraspace = ( V * M * vec4( vertex_pos, 1 ) ).xyz;
	eye_dir_cameraspace = vec3( 0, 0, 0 ) - vertex_pos_cameraspace;
	
	vec3 light_pos_cameraspace = ( V * light_pos_worldspace ).xyz;
	light_dir_cameraspace = light_pos_cameraspace + eye_dir_cameraspace;
	
	normal_cameraspace = ( V * M * vec4( vertex_normal, 0 ) ).xyz;
	
	uv = vertex_uv;
}