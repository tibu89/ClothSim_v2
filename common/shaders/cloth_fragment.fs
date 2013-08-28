#version 330 core

in vec3 eye_dir_cameraspace;
in vec3 light_dir_cameraspace;
in vec3 normal_cameraspace;
in vec2 uv;

out vec3 color;

uniform vec3 my_color;
uniform vec3 light_pos_worldspace;
uniform mat4 M;
uniform mat4 V;

uniform sampler2D normal_map_sampler;
uniform int normal_sign;

mat3 get_cotangent_frame( vec3 N, vec3 pos, vec2 uv )
{
	vec3 dp1 = dFdx( pos );
	vec3 dp2 = dFdy( pos );
	
	vec2 duv1 = dFdx( uv );
	vec2 duv2 = dFdy( uv );
	
	vec3 dp1perp = cross( N, dp1 );
	vec3 dp2perp = cross( dp2, N );
	
	vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
	vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
	
	float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
	
	return mat3( T * invmax, B * invmax, N );
}

vec3 perturb_normal( vec3 N, vec3 V, vec2 tex )
{
	vec3 normal_map = texture2D( normal_map_sampler, tex ).xyz;
	normal_map = normal_map * 2.0f - 1.0f;
	
	mat3 TBN = get_cotangent_frame( N, -V, tex );
	return normalize( TBN * normal_map );
}

void main()
{
	vec3 light_color = vec3( 1, 1, 1 );
	
	vec3 material_diffuse_color = my_color;
	//vec3 material_diffuse_color = texture2D( normal_map_sampler, uv ).xyz;
	vec3 material_ambient_color = vec3( 0.1, 0.1, 0.1 ) * material_diffuse_color;
	
	vec3 n = normalize( normal_cameraspace );
	vec3 v = normalize( eye_dir_cameraspace );
	
	n = perturb_normal( n, v, uv );
	
	vec3 l = normalize( light_dir_cameraspace );
	
	n = normal_sign * n;
	
	float cos_theta = clamp( dot( n, l ), 0, 1 );
	
	color = material_ambient_color + material_diffuse_color * light_color * cos_theta;
	//color = n;
}