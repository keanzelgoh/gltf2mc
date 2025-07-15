#!/usr/bin/env python3
"""
Simple glTF to Minecraft Bedrock JSON Converter
Fixed version that handles buffer loading properly.
"""

import json
import math
import numpy as np
from pygltflib import GLTF2
from typing import List, Dict, Tuple, Optional
import os

class SimpleGLTFToMCBConverter:
    def __init__(self, voxel_size: float = 1.0, max_cubes: int = 1000):
        self.voxel_size = voxel_size
        self.max_cubes = max_cubes
        self.texture_width = 64
        self.texture_height = 64
        
    def load_gltf(self, filepath: str) -> GLTF2:
        """Load a glTF file with proper buffer handling."""
        try:
            print(f"Loading: {filepath}")
            gltf = GLTF2().load(filepath)
            
            # Handle buffer loading based on file type
            if filepath.lower().endswith('.glb'):
                # GLB files have embedded buffers
                print("GLB file detected - buffers should be embedded")
            else:
                # GLTF files may have external buffers
                base_path = os.path.dirname(os.path.abspath(filepath))
                self._load_external_buffers(gltf, base_path)
            
            return gltf
        except Exception as e:
            raise ValueError(f"Failed to load glTF file: {e}")
    
    def _load_external_buffers(self, gltf: GLTF2, base_path: str):
        """Load external buffer files for .gltf files."""
        for i, buffer in enumerate(gltf.buffers):
            if hasattr(buffer, 'uri') and buffer.uri:
                if buffer.uri.startswith('data:'):
                    # Data URI - already embedded
                    continue
                    
                # External file
                buffer_path = os.path.join(base_path, buffer.uri)
                print(f"Looking for buffer: {buffer_path}")
                
                if os.path.exists(buffer_path):
                    try:
                        with open(buffer_path, 'rb') as f:
                            buffer.data = f.read()
                        print(f"✓ Loaded buffer: {buffer.uri} ({len(buffer.data)} bytes)")
                    except Exception as e:
                        print(f"✗ Failed to load buffer {buffer.uri}: {e}")
                else:
                    print(f"✗ Buffer file not found: {buffer_path}")
    
    def extract_vertices(self, gltf: GLTF2) -> List[np.ndarray]:
        """Extract vertex positions from all meshes."""
        all_vertices = []
        
        print(f"Processing {len(gltf.meshes)} meshes...")
        
        for mesh_idx, mesh in enumerate(gltf.meshes):
            print(f"  Mesh {mesh_idx}: {len(mesh.primitives)} primitives")
            
            for prim_idx, primitive in enumerate(mesh.primitives):
                try:
                    vertices = self._extract_primitive_vertices(gltf, primitive)
                    if vertices is not None and len(vertices) > 0:
                        all_vertices.append(vertices)
                        print(f"    Primitive {prim_idx}: {len(vertices)} vertices")
                    else:
                        print(f"    Primitive {prim_idx}: No vertices extracted")
                except Exception as e:
                    print(f"    Primitive {prim_idx}: Error - {e}")
                    continue
        
        return all_vertices
    
    def _extract_primitive_vertices(self, gltf: GLTF2, primitive) -> Optional[np.ndarray]:
        """Extract vertices from a single primitive."""
        if not hasattr(primitive.attributes, 'POSITION'):
            return None
            
        # Get position accessor
        pos_accessor_idx = primitive.attributes.POSITION
        position_accessor = gltf.accessors[pos_accessor_idx]
        
        # Get buffer view
        buffer_view = gltf.bufferViews[position_accessor.bufferView]
        
        # Get buffer
        buffer_obj = gltf.buffers[buffer_view.buffer]
        
        # Check if buffer data is available
        if not hasattr(buffer_obj, 'data') or buffer_obj.data is None:
            print(f"      No buffer data available for accessor {pos_accessor_idx}")
            return None
        
        # Calculate offsets
        buffer_offset = getattr(buffer_view, 'byteOffset', 0)
        accessor_offset = getattr(position_accessor, 'byteOffset', 0)
        total_offset = buffer_offset + accessor_offset
        
        # Calculate data size (3 floats per vertex, 4 bytes per float)
        vertex_count = position_accessor.count
        data_size = vertex_count * 3 * 4
        
        # Extract and convert data
        try:
            vertex_data = buffer_obj.data[total_offset:total_offset + data_size]
            vertices = np.frombuffer(vertex_data, dtype=np.float32).reshape(-1, 3)
            return vertices
        except Exception as e:
            print(f"      Error extracting vertex data: {e}")
            return None
    
    def calculate_bounds(self, vertices_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate bounding box of all vertices."""
        if not vertices_list:
            return np.array([0, 0, 0]), np.array([0, 0, 0])
        
        all_vertices = np.vstack(vertices_list)
        min_bounds = np.min(all_vertices, axis=0)
        max_bounds = np.max(all_vertices, axis=0)
        
        print(f"Bounds: min={min_bounds}, max={max_bounds}")
        return min_bounds, max_bounds
    
    def voxelize_mesh(self, vertices_list: List[np.ndarray]) -> List[Tuple[float, float, float]]:
        """Convert mesh vertices to voxel grid."""
        if not vertices_list:
            return []
        
        min_bounds, max_bounds = self.calculate_bounds(vertices_list)
        
        # Create voxel grid
        voxel_positions = set()
        
        for vertices in vertices_list:
            for vertex in vertices:
                # Convert to voxel coordinates
                voxel_pos = np.floor((vertex - min_bounds) / self.voxel_size).astype(int)
                voxel_positions.add(tuple(voxel_pos))
        
        print(f"Generated {len(voxel_positions)} unique voxel positions")
        
        # Convert back to world coordinates (centered)
        world_positions = []
        center_offset = (max_bounds + min_bounds) / 2
        
        for voxel_pos in voxel_positions:
            world_pos = (
                voxel_pos[0] * self.voxel_size - center_offset[0],
                voxel_pos[1] * self.voxel_size - min_bounds[1],  # Keep bottom at 0
                voxel_pos[2] * self.voxel_size - center_offset[2]
            )
            world_positions.append(world_pos)
        
        # Limit number of cubes
        if len(world_positions) > self.max_cubes:
            print(f"⚠️  Limiting output to {self.max_cubes} cubes (was {len(world_positions)})")
            world_positions = world_positions[:self.max_cubes]
        
        return world_positions
    
    def create_minecraft_geometry(self, model_name: str, cube_positions: List[Tuple[float, float, float]]) -> Dict:
        """Create Minecraft Bedrock geometry JSON."""
        
        # Create cubes
        cubes = []
        uv_x = 0
        uv_y = 0
        uv_step = 16
        
        for i, (x, y, z) in enumerate(cube_positions):
            # Convert to Minecraft coordinates
            mc_x = x * 16  # Scale to Minecraft units
            mc_y = y * 16
            mc_z = z * 16
            
            cube = {
                "origin": [mc_x, mc_y, mc_z],
                "size": [self.voxel_size * 16, self.voxel_size * 16, self.voxel_size * 16],
                "uv": [uv_x, uv_y]
            }
            
            cubes.append(cube)
            
            # Update UV coordinates
            uv_x += uv_step
            if uv_x >= self.texture_width:
                uv_x = 0
                uv_y += uv_step
                if uv_y >= self.texture_height:
                    uv_y = 0
        
        # Create the full geometry structure
        geometry = {
            "format_version": "1.12.0",
            "minecraft:geometry": [
                {
                    "description": {
                        "identifier": f"geometry.{model_name.replace(' ', '_').lower()}",
                        "texture_width": self.texture_width,
                        "texture_height": self.texture_height,
                        "visible_bounds_width": 3,
                        "visible_bounds_height": 3,
                        "visible_bounds_offset": [0, 1, 0]
                    },
                    "bones": [
                        {
                            "name": "root",
                            "pivot": [0, 0, 0],
                            "cubes": cubes
                        }
                    ]
                }
            ]
        }
        
        return geometry
    
    def convert(self, input_path: str, output_path: str, model_name: Optional[str] = None) -> None:
        """Convert glTF file to Minecraft Bedrock JSON."""
        
        print("=" * 50)
        print("glTF to Minecraft Bedrock Converter")
        print("=" * 50)
        
        # Check if input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Load glTF
        gltf = self.load_gltf(input_path)
        
        # Extract vertices
        print("\n🔍 Extracting vertices...")
        vertices_list = self.extract_vertices(gltf)
        
        if not vertices_list:
            raise ValueError("❌ No mesh data found in glTF file")
        
        total_vertices = sum(len(v) for v in vertices_list)
        print(f"✅ Found {len(vertices_list)} meshes with {total_vertices} total vertices")
        
        # Voxelize
        print(f"\n🧊 Converting to voxels (size: {self.voxel_size})...")
        cube_positions = self.voxelize_mesh(vertices_list)
        
        if not cube_positions:
            raise ValueError("❌ No cubes generated from mesh")
        
        print(f"✅ Generated {len(cube_positions)} cubes")
        
        # Generate model name if not provided
        if model_name is None:
            model_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Create Minecraft geometry
        print(f"\n⚙️  Creating Minecraft geometry...")
        geometry = self.create_minecraft_geometry(model_name, cube_positions)
        
        # Save to file
        print(f"\n💾 Saving to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(geometry, f, indent=2)
        
        print("\n" + "=" * 50)
        print("✅ CONVERSION COMPLETE!")
        print("=" * 50)
        print(f"📁 Output file: {output_path}")
        print(f"🏷️  Model identifier: geometry.{model_name.replace(' ', '_').lower()}")
        print(f"🧊 Cubes generated: {len(cube_positions)}")
        print(f"📐 Voxel size: {self.voxel_size}")
        print("=" * 50)

# Usage example
if __name__ == "__main__":
    # Create converter with custom settings
    converter = SimpleGLTFToMCBConverter(
        voxel_size=0.5,    # Smaller = more detail
        max_cubes=2000     # Increase for complex models
    )
    
    # Convert - UPDATE THESE PATHS!
    try:
        converter.convert(
            input_path="Lamborghini Vision 12.gltf",  # Your input file
            output_path="lamborghini.json",           # Output file
            model_name="lamborghini_vision"           # Model name
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure the .gltf file and any .bin files are in the same folder")
        print("2. Try converting your model to .glb format instead")
        print("3. Check that the file path is correct")
        print("4. Try with a simpler glTF model first")
