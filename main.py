import pygame
import sys
import time
import random
import datetime
from pygame.locals import *
from Primitives import *
from RGB import RGB
from Material import Material
import taichi as ti
from Vector import *

ti.init(arch=ti.gpu)
vec3_type = ti.types.vector(3, ti.f32)
ray_type = ti.types.struct(origin=ti.types.vector(3, ti.f32), dest=ti.types.vector(3, ti.f32))
material_type = ti.types.struct(color=ti.types.vector(3, ti.f32), opacity=ti.f32, reflect=ti.f32, luma=ti.f32)


background_color = RGB(0, 0, 0)
xres = 640
yres = 480
psize = 1
recurse_limit = 20
Viewportz = 100

@ti.func
def distance(one: ti.types.vector(3, ti.f32), two: ti.types.vector(3, ti.f32)) -> float:
    return math.sqrt((one[0] - two[0]) ** 2 + (one[1] - two[1]) ** 2 + (one[2] - two[2]) ** 2)

@ti.func
def lighting(hit) -> ti.types.vector(3, ti.f32):
    # How much light things get from cosine shading.
    diffuse_coefficient = 1.
    shade = vec_mult_vec(vec3_type((0, 0, 1)), hit.normal)
    if shade < 0.:
        shade = 0.

    point_color = hit.object.material.color * (diffuse_coefficient * shade)
    if hit.object.material.luma > 0:
        retval = hit.object.material.color
    else:
        retval = point_color
    return retval

@ti.func
def raytrace(cast_ray: ray_type, r: ti.uint8 = recurse_limit) -> ti.types.vector(3, ti.f32):
    hit_distance = 0.
    hit = 0.
    for thing in World:
        if intersection := thing.hit(cast_ray):
            t_hit_distance = distance(cast_ray.origin, intersection.hit_point)
            if t_hit_distance < hit_distance or hit_distance == 0.:
                hit_distance = t_hit_distance
                hit = intersection
    if not hit_distance:
        return background_color
    # At this point, hit.object is the closest item that ray intersected with.
    # Check for end of recursion
    if r <= 1:
        # Now we apply lighting
        return lighting(hit)

    # Check for Reflectance
    if hit.object.material.reflect > 0:
        reflect_ray = ray_type(origin=hit.hit_point, dest=vec_normalize(cast_ray.dest + (2 * hit.normal * (0 - (hit.normal * cast_ray.dest)))))
        reflect_color = raytrace(reflect_ray, r - 1)
        return lighting(hit) + (reflect_color * hit.object.material.reflect)
    else:
        # The object isn't reflective.
        return lighting(hit)


@ti.kernel
def render() -> ti.field(dtype=int, shape=(xres, yres)):
    pixels = ti.field(dtype=int, shape=(xres, yres))
    ray = ray_type(dest=vec3_type((0, 0, -1)), origin=vec3_type((0, 0, 0)))
    #  Hardcoded Viewport for now
    ray.origin[2] = Viewportz

    for x, y in pixels:
        ray.origin[1] = psize * (y - 0.5 * (yres - 1))
        ray.origin[0] = psize * (x - 0.5 * (xres - 1))
        pixels[x, y] = raytrace(ray).finalcolor()
    return pixels


def main():
    pygame.init()
    window_surface_obj = pygame.display.set_mode((xres, yres))
    display = pygame.PixelArray(window_surface_obj)
    starttime = time.time()
    pygame.display.set_caption("PyTrace - Render in Progress...")
    render(display)
    pygame.display.set_caption(f"PyTrace Render Finished - Total time: {time.time() - starttime} seconds")
    print(f'Time : {time.time() - starttime}')

    pygame.event.set_allowed(pygame.QUIT)

    filename = datetime.datetime.strftime(datetime.datetime.now(), "%H.%M.%S_%d-%b-%Y") + '.png'
    pygame.image.save(window_surface_obj, filename)
    pygame.event.set_allowed((pygame.QUIT, pygame.KEYDOWN))
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.event.post(pygame.event.Event(QUIT))


World = []
random.seed()

# World.append(Cube(Point3D(0,0,0),Point3D(-100,100,0),Point3D(-100,0,-100),Material(RGB(.5,0,0))))


World.append(Sphere(Point3D(50, -50, 1), 95, Material(RGB(.5, 0, 0), 1.0, 0.7, 0.0)))
World.append(Sphere(Point3D(-50, -50, 1), 95, Material(RGB(0, .5, 0), 1.0, 0.7, 0.0)))
World.append(Sphere(Point3D(1, 50, 1), 95, Material(RGB(0, 0, .5), 1.0, 0.7, 0.0)))

# World.append(Sphere(Point3D(-260, -100, 0), 40, Material(RGB(1.0, 1.0, 1.0), 1.0, 0.0, 1.0)))
# World.append(Plane(Point3D(1, 90, 200), Vector3D(0, 0, -1), Material(RGB(.3, .3, .3), 1.0, 0.0, 0.0)))

# White Sphere on Right - should be merged with red sphere below
# World.append(Sphere(Point3D(180, 0, 0), 30, Material(RGB(255, 255, 255), 1.0, 0.0, 0.0)))

# Brown Sphere behind Red Sphere
# World.append(Sphere(Point3D(150, 0, -10), 30, Material(RGB(255, 255, 0), 1.0, 0.0, 0.0)))

# Red Sphere on Right
# World.append(Sphere(Point3D(150, 0, 0), 30, Material(RGB(255, 0, 0), 1.0, 0.0, 0.0)))

# Dim Red Sphere on Left
# World.append(Sphere(Point3D(-150, 0, 0), 30, Material(RGB(127, 0, 0), 1.0, 0.0, 0.0)))

# Green sphere on bottom
# World.append(Sphere(Point3D(0, 150, 0), 30, Material(RGB(0, 255, 0), 1.0, 0.0, 0.0)))

# Dim green sphere on top
# World.append(Sphere(Point3D(1, -150, 1), 30, Material(RGB(0, 127, 0), 1.0, 1.0, 0.0)))

# Bright Blue sphere on bottom right
# World.append(Sphere(Point3D(150, 150, 0), 30, Material(RGB(0, 0, 255), 1.0, 0.0, 0.0)))

# Dim Blue Sphere on top left
# World.append(Sphere(Point3D(-150, -150, 0), 30, Material(RGB(0, 0, 127), 1.0, 0.0, 0.0)))

# add a light
# World.append(Sphere(Point3D(0, 0, 0), 30, Material(RGB(255, 255, 255), 1.0, 1.0, 1.0)))

main()
