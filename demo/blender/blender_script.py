from os import system
import bpy

file_path = __file__.replace('blender_script.py', '')
print(file_path)
bpy.ops.render.render(layer='View Layer')
print(system(f"cd {file_path} && python ..\omt_image_utils_backup.py"))