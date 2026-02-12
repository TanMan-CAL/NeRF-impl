# utils/utils_vec3.py
import math
import random

class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, t):
        if isinstance(t, Vec3):
            return Vec3(self.x * t.x, self.y * t.y, self.z * t.z)
        return Vec3(self.x * t, self.y * t, self.z * t)
    
    def __rmul__(self, t):
        return self * t
    
    def __truediv__(self, t):
        return Vec3(self.x / t, self.y / t, self.z / t)
    
    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)
    
    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        elif i == 2:
            return self.z
    
    def __repr__(self):
        return f"Vec3({self.x}, {self.y}, {self.z})"
    
    def length_squared(self):
        return self.x * self.x + self.y * self.y + self.z * self.z
    
    def length(self):
        return math.sqrt(self.length_squared())
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def unit_vector(self):
        return self / self.length()
    
    def near_zero(self):
        s = 1e-8
        return abs(self.x) < s and abs(self.y) < s and abs(self.z) < s
    
    @staticmethod
    def random(min_val=0.0, max_val=1.0):
        return Vec3(
            random.uniform(min_val, max_val),
            random.uniform(min_val, max_val),
            random.uniform(min_val, max_val)
        )
    
    @staticmethod
    def random_in_unit_sphere():
        while True:
            p = Vec3.random(-1, 1)
            if p.length_squared() < 1:
                return p
    
    @staticmethod
    def random_unit_vector():
        return Vec3.random_in_unit_sphere().unit_vector()
    
    @staticmethod
    def random_in_unit_disk():
        while True:
            p = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), 0)
            if p.length_squared() < 1:
                return p
    
    @staticmethod
    def random_in_hemisphere(normal):
        in_unit_sphere = Vec3.random_in_unit_sphere()
        if in_unit_sphere.dot(normal) > 0.0:  # same hemisphere as the normal
            return in_unit_sphere
        else:
            return -in_unit_sphere
        
Color = Vec3
