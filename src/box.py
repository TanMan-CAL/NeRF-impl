# src/box.py
from utils.utils_vec3 import Vec3
INFINITY = float('inf')

class Box():
    def __init__(self, min_point, max_point, material):
        self.min_point = min_point
        self.max_point = max_point
        self.material = material
    
    def hit(self, r, ray_t, rec):
        t_min = (self.min_point.x - r.origin.x) / r.direction.x if r.direction.x != 0 else -INFINITY
        t_max = (self.max_point.x - r.origin.x) / r.direction.x if r.direction.x != 0 else INFINITY
        
        if r.direction.x < 0:
            t_min, t_max = t_max, t_min
        
        t_y_min = (self.min_point.y - r.origin.y) / r.direction.y if r.direction.y != 0 else -INFINITY
        t_y_max = (self.max_point.y - r.origin.y) / r.direction.y if r.direction.y != 0 else INFINITY
        
        if r.direction.y < 0:
            t_y_min, t_y_max = t_y_max, t_y_min
        
        if t_min > t_y_max or t_y_min > t_max:
            return False
        
        if t_y_min > t_min:
            t_min = t_y_min
        
        if t_y_max < t_max:
            t_max = t_y_max
        
        t_z_min = (self.min_point.z - r.origin.z) / r.direction.z if r.direction.z != 0 else -INFINITY
        t_z_max = (self.max_point.z - r.origin.z) / r.direction.z if r.direction.z != 0 else INFINITY
        
        if r.direction.z < 0:
            t_z_min, t_z_max = t_z_max, t_z_min
        
        if t_min > t_z_max or t_z_min > t_max:
            return False
        
        if t_z_min > t_min:
            t_min = t_z_min
        
        if t_z_max < t_max:
            t_max = t_z_max
        
        if not ray_t.surrounds(t_min) and not ray_t.surrounds(t_max):
            return False
        
        t = t_min if ray_t.surrounds(t_min) else t_max
        
        rec.t = t
        rec.p = r.at(t)
        
        if abs(rec.p.x - self.min_point.x) < 1e-8:
            outward_normal = Vec3(-1, 0, 0)
        elif abs(rec.p.x - self.max_point.x) < 1e-8:
            outward_normal = Vec3(1, 0, 0)
        elif abs(rec.p.y - self.min_point.y) < 1e-8:
            outward_normal = Vec3(0, -1, 0)
        elif abs(rec.p.y - self.max_point.y) < 1e-8:
            outward_normal = Vec3(0, 1, 0)
        elif abs(rec.p.z - self.min_point.z) < 1e-8:
            outward_normal = Vec3(0, 0, -1)
        else:
            outward_normal = Vec3(0, 0, 1)
        
        rec.set_face_normal(r, outward_normal)
        rec.material = self.material
        
        return True