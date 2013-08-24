#version 330 core

layout( location = 0 ) in vec3 vertex_pos;
layout( location = 1 ) in vec3 normal;

out vec3 fragmentColor;

uniform mat4 MVP;
uniform mat4 V;
uniform mat4 M;
uniform vec3 my_color;
uniform vec4 light_pos_world;

void main()
{
	gl_Position = MVP * vec4( vertex_pos, 1 );
	
	fragmentColor = my_color;
}