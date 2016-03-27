//=============================================================================================

// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Hegedues Daniel
// Neptun : YBY8BK
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 130
    precision highp float;

				uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

				in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
	in vec3 vertexColor;	    // variable input from Attrib Array selected by glBindAttribLocation
	out vec3 color;				// output attribute

				void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 130
    precision highp float;

				in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

				void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};

// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}
};


struct vec2 {
	float x,y;

	vec2() {}

	vec2(float argx, float argy) {
		x = argx;
		y = argy;
	}

	vec2 operator+(const vec2 arg) {
		vec2 v;
		v.x = this->x + arg.x;
		v.y = this->y + arg.y;
		return v;
	}
	vec2 operator-(const vec2 arg) {
		vec2 v;
		v.x = this->x - arg.x;
		v.y = this->y - arg.y;
		return v;
	}
	vec2 operator/(const float arg) {
		vec2 v;
		v.x = this->x / arg;
		v.y = this->y / arg;
		return v;
	}
	vec2 operator*(const float arg) {
		vec2 v;
		v.x = this->x * arg;
		v.y = this->y * arg;
		return v;
	}
	vec2& operator=(const vec2 arg) {
		this->x = arg.x;
		this->y = arg.y;
		return *this;
	}
};

// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
	bool star;
public:
	Camera() {
		Animate(0, 0, 0);
		star = false;
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void toggleStar(bool b) {
		star = b;
	}

	bool getStarState() {
		return star;
	}

	void Animate(float t, float starX, float starY) {
		wWx = 40;
		wWy = 40;
		if(star) {
			wCx = starX;
			wCy = starY;
		} else {
			wCx = 0; // 10 * cosf(t);
			wCy = 0;
		}
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

class LineStrip {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[500000]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
public:
	LineStrip() {
		nVertices = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void AddPoint(float cX, float cY) {
		// fill interleaved data
		vertexData[5 * nVertices] = cX;
		vertexData[5 * nVertices + 1] = cY;
		vertexData[5 * nVertices + 2] = 1; // red
		vertexData[5 * nVertices + 3] = 1; // green
		vertexData[5 * nVertices + 4] = 0; // blue
		nVertices++;

		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	int getVertexCount() {
		return nVertices;
	}

	void removeAll() {
		nVertices = 0;
	}

	void Draw() {
		if (nVertices > 2) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nVertices);
		}
	}
};

class ControlPoint {
	vec2 pos;
	float t;
	vec2 vv;
public:
	ControlPoint() {}

	ControlPoint(float argt, float x, float y) {
		pos = vec2(x, y);
		t = argt;
		vv = vec2(0,0);
	}

	vec2 getPos() {
		return pos;
	}

	float getT() {
		return t;
	}

	vec2 getVV() {
		return vv;
	}

	void setT(float arg) {
		t = arg;
	}

	void setPos(vec2 v) {
		pos = v;
	}

	void setVV(vec2 vv) {
		this->vv = vv;
	}

	ControlPoint& operator=(ControlPoint arg) {
		this->pos = arg.getPos();
		this->t = arg.getT();
		this->vv = arg.getVV();
		return *this;
	}

};

class CatmullRom {
	ControlPoint cps[20];
	int n;
public:
	CatmullRom() {
		n = 0;
	}

	void add(float t, float argx, float argy, LineStrip &l) {
		vec4 wVertex = vec4(argx, argy, 0, 1) * camera.Pinv() * camera.Vinv();

		float x = wVertex.v[0];
		float y = wVertex.v[1];

		ControlPoint cp = ControlPoint(t, x, y);
		cps[n] = cp;

		for (int i = 1; i < n-1; i++) {
			cps[i].setVV(( ( (cps[i+1].getPos() - cps[i].getPos()) /
											 (cps[i+1].getT() - cps[i].getT()) )
											  								+
										 ( (cps[i].getPos() - cps[i-1].getPos()) /
									 	 (cps[i].getT() - cps[i-1].getT()) ) )
										 								* 0.9f);
		}

		cps[0].setVV(( ( (cps[0+1].getPos() - cps[0].getPos()) / (cps[0+1].getT() - cps[0].getT()) ) + ( (cps[0].getPos() - cps[n].getPos()) / (cps[0].getT() - cps[n].getT()) ) ) * 0.9f);

		cps[n].setVV(( ( (cps[0].getPos() - cps[n].getPos()) / (cps[n].getT()+0.5 - cps[n].getT()) ) + ( (cps[n].getPos() - cps[n-1].getPos()) / (cps[n].getT() - cps[n-1].getT()) ) ) * 0.9f);

		cps[n+1] = cps[0];
		cps[n+1].setT(t + 0.5);

		l.removeAll();
		float dt = 0.025;
		if(n <= 1) {
			l.AddPoint(x, y);
		} else {
			for(float i = cps[0].getT(); i <= cps[n+1].getT(); i += dt ) {
				vec2 h = r(i);
				l.AddPoint(h.x, h.y);
			}
		}
		n++;
	}

	vec2 hermite(float t, ControlPoint cV, ControlPoint nV) {
		vec2 a0 = cV.getPos();
		vec2 a1 = cV.getVV();
		vec2 a2 = (((nV.getPos() - cV.getPos())*3) / powf((nV.getT() - cV.getT()), 2.0)) - ((nV.getVV() + (cV.getVV() * 2)) / (nV.getT() - cV.getT()));
		vec2 a3 = (((cV.getPos() - nV.getPos())*2) / powf((nV.getT() - cV.getT()), 3.0)) + ((nV.getVV() + cV.getVV()) / powf((nV.getT() - cV.getT()), 2.0));

		vec2 coord = a3 * powf((t - cV.getT()), 3.0) + a2 * powf((t - cV.getT()), 2.0) + a1 * (t - cV.getT()) + a0;

		return coord;
	}

	vec2 r(float t) {
		for(int i = 0; i <= n+1; i++) {
			if( cps[i].getT() <= t && t <= cps[i+1].getT() ) {
				vec2 h = hermite(t, cps[i], cps[i+1]);
				return h;
			}
		}
		return vec2(0,0);
	}

	ControlPoint* getCps() {
		return cps;
	}

	int getSize() {
		return n;
	}
};

const float grav = 0.03f;
const float mu = 0.007f;

class Star {
	unsigned int vao;	// vertex array object id
	float sx, sy;
	float wTx, wTy;
	float rad;
	vec2 v;
public:
	Star() {
		Animate(0, NULL, NULL);
		v = vec2(0,0);
	}

	float getwTx() { return wTx; }
	float getwTy() { return wTy; }

	void Create(float x, float y, float r, float red, float green, float blue) {
		wTx = x;
		wTy = y;
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
											   //static float vertexCoords[6] = coordArray;	// vertex data on the CPU
		static float vertexCoords[32];
		float* coordArray = generateCoords(0, 0, r);
		for (int i = 0; i < 32; ++i) {
			vertexCoords[i] = coordArray[i];
		}

		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords),  // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[48];	// vertex data on the CPU
		int i = 0;
		while (i < 48) {
			vertexColors[i] = red;
			vertexColors[i + 1] = green;
			vertexColors[i + 2] = blue;
			i = i + 3;
		}
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	float* generateCoords(float x, float y, float r) {
		float* coordArray = new float[32];
		float rad = (float)M_PI;
		float currentX = x + r * cosf(rad);
		float currentY = y + r * sinf(rad);
		int i = 0;
		while (i < 32) {
			coordArray[i] = currentX;
			coordArray[i + 1] = currentY;

			rad += (float)(2 * M_PI / 8);

			float nextX = x + r * cosf(rad);
			float nextY = y + r * sinf(rad);
			float d = (float)sqrt((currentX - nextX)*(currentX - nextX) + (currentY - nextY)*(currentY - nextY));
			float tmpX = (currentX + nextX) / 2;
			float tmpY = (currentY + nextY) / 2;
			float vecX = tmpX - x;
			float vecY = tmpY - y;
			tmpX += vecX * d;
			tmpY += vecY * d;

			coordArray[i + 2] = tmpX;
			coordArray[i + 3] = tmpY;
			currentX = nextX;
			currentY = nextY;

			i = i + 4;
		}

		return coordArray;
	}

	void Animate(float t, Star* mStar, CatmullRom* c) {
		if (c != NULL) {
			int n = c->getSize();
			if(n >= 3) {
				float modt = fmodf(t/10, c->getCps()[n].getT() - c->getCps()[0].getT()) + c->getCps()[0].getT();
				wTx = c->r(modt).x;
				wTy = c->r(modt).y;
			}
		}
		float r;
		float gravF;
		if (mStar != NULL) {
			r = sqrtf(((mStar->getwTx() - wTx) * (mStar->getwTx() - wTx)) + ((mStar->getwTy() - wTy) * (mStar->getwTy() - wTy)));
			if (fabsf(r) <= 0.5) {
				gravF = 0;
			}
			else {
				gravF = grav / (r*r);
			}

			float vecX = mStar->getwTx() - wTx;
			float vecY = mStar->getwTy() - wTy;
			vec2 vec = vec2(vecX / r, vecY / r);
			vec2 acceleration = vec * gravF;
			vec2 fricV = v * (-mu);
			acceleration = acceleration + fricV;
			v = v + acceleration;
			wTx = wTx + v.x;
			wTy = wTy + v.y;
		}

		sx = 1.0f + 0.2f * cosf(t);
		sy = 1.0f + 0.2f * cosf(t);

		if (t == 0) {
			rad = 0;
			wTx = 1 * wTx;
			wTy = 1 * wTy;
		}
		rad = t;
	}

	void Draw() {

		mat4 M(cosf(rad) + sx, sinf(rad), 0, 0,
			-sinf(rad), cosf(rad) + sy, 0, 0,
			0, 0, 1, 0,
			wTx, wTy, 0, 1); // model matrix

		mat4 MVPTransform = M * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 16);	// draw a single triangle with vertices defined in vao
	}
};

// The virtual world: collection of two objects
Star mainStar;
Star star2;
Star star3;
CatmullRom cmr;
LineStrip lineStrip;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	mainStar.Create(-10, 10, 1, 212.0f/255.0f, 1.0f, 0.0f);
	star2.Create(    -8, -8, 1, 179.0f/255.0f, 149.0f/255.0f, 0.0f/255.0f);
	star3.Create(     6, -9, 1, 178.0f/255.0f, 148.0f/255.0f, 0.1f/255.0f);
	lineStrip.Create();

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect Attrib Arrays to input variables of the vertex shader
	glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
	glBindAttribLocation(shaderProgram, 1, "vertexColor");    // vertexColor gets values from Attrib Array 1

															  // Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	mainStar.Draw();
	star2.Draw();
	star3.Draw();
	lineStrip.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		if(camera.getStarState()) {
			camera.toggleStar(false);
		} else {
			camera.toggleStar(true);
		}
	}

	glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	long time = glutGet(GLUT_ELAPSED_TIME);
	float sec = time / 1000.0f;
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		cmr.add(sec, cX, cY, lineStrip);
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	mainStar.Animate(sec, NULL, &cmr);					// animate the triangle object
	camera.Animate(sec, mainStar.getwTx(), mainStar.getwTy());					// animate the camera
	star2.Animate(sec, &mainStar, NULL);
	star3.Animate(sec, &mainStar, NULL);
	glutPostRedisplay();					// redraw the scene
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
