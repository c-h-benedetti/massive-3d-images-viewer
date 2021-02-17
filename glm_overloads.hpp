#ifndef OVERLOADING_COUT_GLM_INCLUDED
#define OVERLOADING_COUT_GLM_INCLUDED

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
// ~      OVERLOADING OF COUT FOR VEC3 & VEC4                                                  ~
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

ostream& operator<<(ostream& o, glm::vec3& v){
    o << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return o;
}

ostream& operator<<(ostream& o, glm::ivec3& v){
    o << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return o;
}

ostream& operator<<(ostream& o, glm::vec4& v){
    o << "(" << v.x << ", " << v.y << ", " << v.z  << ", " << v.w << ")";
    return o;
}

ostream& operator<<(ostream& o, glm::mat4& m){
    o << "[" << m[0][0] << " " << m[1][0] << " " << m[2][0] << " " << m[3][0] << std::endl << m[0][1] << " " << m[1][1] << " " << m[2][1] << " " << m[3][1] << std::endl << m[0][2] << " " << m[1][2] << " " << m[2][2] << " " << m[3][2] << std::endl << m[0][3] << " " << m[1][3] << " " << m[2][3] << " " << m[3][3] << "]";
    return o;
}

#endif //OVERLOADING_COUT_GLM_INCLUDED
