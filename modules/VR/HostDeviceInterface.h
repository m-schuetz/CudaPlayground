
#pragma once

#include "builtin_types.h"

struct mat4{
	float4 rows[4];

	static mat4 identity(){
		mat4 id;

		id.rows[0] = {1.0f, 0.0f, 0.0f, 0.0f};
		id.rows[1] = {0.0f, 1.0f, 0.0f, 0.0f};
		id.rows[2] = {0.0f, 0.0f, 1.0f, 0.0f};
		id.rows[3] = {0.0f, 0.0f, 0.0f, 1.0f};

		return id;
	}

	static mat4 rotate(float angle, float3 axis){
		// see https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

		float cosa = cos(-angle);
		float sina = sin(-angle);

		float ux = axis.x;
		float uy = axis.y;
		float uz = axis.z;

		mat4 rot;
		rot.rows[0].x = cosa + ux * ux * (1.0f - cosa);
		rot.rows[0].y = ux * uy * (1.0f - cosa) - uz * sina;
		rot.rows[0].z = ux * uz * (1.0f - cosa) + uy * sina;
		rot.rows[0].w = 0.0f;

		rot.rows[1].x = uy * ux * (1.0f - cosa) + uz * sina;
		rot.rows[1].y = cosa + uy * uy * (1.0f - cosa);
		rot.rows[1].z = uy * uz * (1.0f - cosa) - ux * sina;
		rot.rows[1].w = 0.0f;

		rot.rows[2].x = uz * ux * (1.0f - cosa) - uy * sina;
		rot.rows[2].y = uz * uy * (1.0f - cosa) + ux * sina;
		rot.rows[2].z = cosa + uz * uz * (1.0f - cosa);
		rot.rows[2].w = 0.0f;

		rot.rows[3].x = 0.0f;
		rot.rows[3].y = 0.0f;
		rot.rows[3].z = 0.0f;
		rot.rows[3].w = 1.0f;

		return rot;
	}

	static mat4 translate(float x, float y, float z){

		mat4 trans = mat4::identity();

		trans.rows[0].w = x;
		trans.rows[1].w = y;
		trans.rows[2].w = z;

		return trans;
	}

	static mat4 scale(float sx, float sy, float sz){

		mat4 scaled = mat4::identity();

		scaled.rows[0].x = sx;
		scaled.rows[1].y = sy;
		scaled.rows[2].z = sz;

		return scaled;
	}

	mat4 transpose(){
		mat4 result;

		result.rows[0] = {rows[0].x, rows[1].x, rows[2].x, rows[3].x};
		result.rows[1] = {rows[0].y, rows[1].y, rows[2].y, rows[3].y};
		result.rows[2] = {rows[0].z, rows[1].z, rows[2].z, rows[3].z};
		result.rows[3] = {rows[0].w, rows[1].w, rows[2].w, rows[3].w};

		return result;
	}

	static mat4 fromRows(float4 row0, float4 row1, float4 row2, float4 row3){
		mat4 result;

		result.rows[0] = row0;
		result.rows[1] = row1;
		result.rows[2] = row2;
		result.rows[3] = row3;

		return result;
	}

	// from glm, func_matrix.inl
	mat4 inverse(){

		// float Coef00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		// float Coef02 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
		// float Coef03 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

		// float Coef04 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		// float Coef06 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
		// float Coef07 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

		// float Coef08 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		// float Coef10 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
		// float Coef11 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

		// float Coef12 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		// float Coef14 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
		// float Coef15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

		// float Coef16 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		// float Coef18 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
		// float Coef19 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

		// float Coef20 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
		// float Coef22 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
		// float Coef23 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

		mat4 m = this->transpose();

		float Coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
		float Coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
		float Coef03 = m[1].z * m[2].w - m[2].z * m[1].w;

		float Coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
		float Coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
		float Coef07 = m[1].y * m[2].w - m[2].y * m[1].w;

		float Coef08 = m[2].y * m[3].x - m[3].y * m[2].z;
		float Coef10 = m[1].y * m[3].x - m[3].y * m[1].z;
		float Coef11 = m[1].y * m[2].x - m[2].y * m[1].z;

		float Coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
		float Coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
		float Coef15 = m[1].x * m[2].w - m[2].x * m[1].w;

		float Coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
		float Coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
		float Coef19 = m[1].x * m[2].z - m[2].x * m[1].z;

		float Coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
		float Coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
		float Coef23 = m[1].x * m[2].y - m[2].x * m[1].y;


		float4 Fac0 = {Coef00, Coef00, Coef02, Coef03};
		float4 Fac1 = {Coef04, Coef04, Coef06, Coef07};
		float4 Fac2 = {Coef08, Coef08, Coef10, Coef11};
		float4 Fac3 = {Coef12, Coef12, Coef14, Coef15};
		float4 Fac4 = {Coef16, Coef16, Coef18, Coef19};
		float4 Fac5 = {Coef20, Coef20, Coef22, Coef23};

		float4 Vec0 = {m[1].x, m[0].x, m[0].x, m[0].x};
		float4 Vec1 = {m[1].z, m[0].z, m[0].z, m[0].z};
		float4 Vec2 = {m[1].y, m[0].y, m[0].y, m[0].y};
		float4 Vec3 = {m[1].w, m[0].w, m[0].w, m[0].w};

		//float4 Inv0 = {Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2};
		//float4 Inv1 = {Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4};
		//float4 Inv2 = {Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5};
		//float4 Inv3 = {Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5};

		//float4 SignA = {+1, -1, +1, -1};
		//float4 SignB = {-1, +1, -1, +1};
		//mat4 Inverse = mat4::fromRows(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB);
		//
		//// printf("%f \n", Inverse.rows[0].y);

		//float4 Row0 =  {Inverse[0].x, Inverse[1].x, Inverse[2].x, Inverse[3].x};

		//float4 Dot0 = m[0] * Row0;
		//float Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

		//float OneOverDeterminant = 1.0f / Dot1;

		//return Inverse * OneOverDeterminant;

		printf("Coef00 = %f \n", Coef00);

		printf("==============\n");
		printf("Coef00 = %f \n", Coef00);
		printf("Coef02 = %f \n", Coef02);
		printf("Coef03 = %f \n", Coef03);
		printf("Coef04 = %f \n", Coef04);
		printf("Coef06 = %f \n", Coef06);
		printf("Coef07 = %f \n", Coef07);
		printf("Coef08 = %f \n", Coef08);
		printf("Coef10 = %f \n", Coef10);
		printf("Coef11 = %f \n", Coef11);
		printf("Coef12 = %f \n", Coef12);
		printf("Coef14 = %f \n", Coef14);
		printf("Coef15 = %f \n", Coef15);
		printf("Coef16 = %f \n", Coef16);
		printf("Coef18 = %f \n", Coef18);
		printf("Coef19 = %f \n", Coef19);
		printf("Coef20 = %f \n", Coef20);
		printf("Coef22 = %f \n", Coef22);
		printf("Coef23 = %f \n", Coef23);

		return *this;
	}

	float4& operator[](int index){
		return rows[index];
	}

	mat4& operator*(float factor){
		
		rows[0].x *= factor;
		rows[0].y *= factor;
		rows[0].z *= factor;
		rows[0].w *= factor;

		rows[1].x *= factor;
		rows[1].y *= factor;
		rows[1].z *= factor;
		rows[1].w *= factor;

		rows[2].x *= factor;
		rows[2].y *= factor;
		rows[2].z *= factor;
		rows[2].w *= factor;

		rows[3].x *= factor;
		rows[3].y *= factor;
		rows[3].z *= factor;
		rows[3].w *= factor;

		return *this;
	}
};



int COLORMODE_TEXTURE          = 0;
int COLORMODE_UV               = 1;
int COLORMODE_TRIANGLE_ID      = 2;
int COLORMODE_TIME             = 3;
int COLORMODE_TIME_NORMALIZED  = 4;
int COLORMODE_VERTEXCOLOR      = 5;

int SAMPLEMODE_NEAREST     = 0;
int SAMPLEMODE_LINEAR      = 1;


struct ControllerState{
	uint32_t packetNum;
	uint64_t buttonPressedMask;
	uint64_t buttonTouchedMask;
};

struct Uniforms{
	float width;
	float height;
	float time;
	mat4 world;
	mat4 view;
	mat4 proj;
	mat4 view_inv;
	mat4 proj_inv;
	mat4 transform;
	int colorMode;
	int sampleMode;

	bool vrEnabled;

	float vr_left_width;
	float vr_left_height;
	mat4 vr_left_world;
	mat4 vr_left_view;
	mat4 vr_left_proj;
	mat4 vr_left_view_inv;
	mat4 vr_left_proj_inv;
	mat4 vr_left_transform;
	bool vr_left_controller_active;
	mat4 vr_left_controller_pose;
	ControllerState vr_left_controller_state;
	

	float vr_right_width;
	float vr_right_height;
	mat4 vr_right_world;
	mat4 vr_right_view;
	mat4 vr_right_proj;
	mat4 vr_right_view_inv;
	mat4 vr_right_proj_inv;
	mat4 vr_right_transform;
	bool vr_right_controller_active;
	mat4 vr_right_controller_pose;
	ControllerState vr_right_controller_state;


};
