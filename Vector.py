from __future__ import annotations
from Normal import Normal
from typing import Optional, Union, Iterable
import math
import taichi as ti

vec3 = ti.types.vector(3, ti.f32)

@ti.func
def vec_add(one: ti.types.vector(3, ti.f32), two: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    return vec3((one[0] + two[0], one[1] + two[1], one[2] + two[2]))

@ti.func
def vec_sub(one: ti.types.vector(3, ti.f32), two: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    return vec3((one[0] - two[0], one[1] - two[1], one[2] - two[2]))

@ti.func
def vec_cross(one: ti.types.vector(3, ti.f32), two: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    return vec3(((one[1] * two[2]) - (one[2] * two[1]), (one[2] * two[0]) - (one[0] * two[2]), (one[0] * two[1]) - (one[1] * two[0])))

@ti.func
def vec_dot(one: ti.types.vector(3, ti.f32), two: ti.types.vector(3, ti.f32)) -> ti.f32:
    return one[0] * two[0] + one[1] * two[1] + one[2] * two[2]

@ti.func
def vec_mult_vec(one: ti.types.vector(3, ti.f32), two: ti.types.vector(3, ti.f32)) -> ti.f32:
    return one[0] * two[0] + one[1] * two[1] + one[2] * two[2]

@ti.func
def vec_div(one: ti.types.vector(3, ti.f32), two: ti.f32) -> ti.types.vector(3, ti.f32):
    return vec3((one[0] / two, one[1] / two, one[2] / two[2]))

@ti.func
def vec_len(one: ti.types.vector(3, ti.f32)) -> float:
    return math.sqrt((one[0] * one[0]) + (one[1] * one[1]) + (one[2] * one[2]))

@ti.func
def vec_normalize(one: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    length = vec_len(one)
    return vec3((one[0] / length, one[1] / length, one[2] / length))



#
# class Vector3D:
#     x: Union[ti.f32, ti.types.vector(3, ti.f32)]
#     y: Optional[ti.f32]
#     z: Optional[ti.f32]
#
#     def __init__(self, x: ti.f32 = 0., y: ti.f32 = 0., z: ti.f32 = 0.) -> None:
#         self.vec = ti.types.vector(3, ti.f32)
#         try:
#             self.vec[iter(x)]
#         except TypeError:
#             self.vec[0] = x
#             self.vec[1] = y
#             self.vec[2] = z
#
#     # ^ - Cross Product
#     def __xor__(self, vector) -> Vector3D:
#         return Vector3D(x=(self.vec[1] * vector.z) - (self.vec[2] * vector.y),
#                         y=(self.vec[2] * vector.x) - (self.vec[0] * vector.z),
#                         z=(self.vec[0] * vector.y) - (self.vec[1] * vector.x))
#
#     def __add__(self, vector) -> Vector3D:
#         return Vector3D(x=vector.x + self.vec[0], y=vector.y + self.vec[1], z=vector.z + self.vec[2])
#
#     def __sub__(self, vector) -> Vector3D:
#         return Vector3D(x=self.vec[0] - vector.x, y=self.vec[1] - vector.y, z=self.vec[2] - vector.z)
#
#     # % - Dot Product
#     def __mod__(self, vector) -> float:
#         return self.vec[0] * vector.x + self.vec[1] * vector.y + self.vec[2] * vector.z
#
#     def __mul__(self, other) -> Union[float, Vector3D]:
#         if isinstance(other, (Vector3D, Normal)):
#             retval = self.vec[0] * other.x + self.vec[1] * other.y + self.vec[2] * other.z
#         else:
#             retval = Vector3D(self.vec[0] * other, self.vec[1] * other, self.vec[2] * other)
#         return retval
#
#     def __rmul__(self, other):
#         return self.__mul__(other)
#
#     def __truediv__(self, other) -> Vector3D:
#         if other == 0:
#             retval = False
#         else:
#             retval = Vector3D(self.vec[0] / other, self.vec[1] / other, self.vec[2] / other)
#         return retval
#
#     def normalize(self) -> Vector3D:
#         length = self.length
#         return Vector3D(self.vec[0] / length, self.vec[1] / length, self.vec[2] / length)
#
#     def __eq__(self, vector) -> bool:
#         return (self.vec[0] == vector.x) and (self.vec[1] == vector.y) and (self.vec[2] == vector.z)
#
#     @property
#     def length(self) -> float:
#         return math.sqrt((self.vec[0] * self.vec[0]) + (self.vec[1] * self.vec[1]) + (self.vec[2] * self.vec[2]))
#
#     def __repr__(self) -> str:
#         return f"Vector3D(x={self.vec[0]}, y={self.vec[1]}, z={self.vec[2]})"
