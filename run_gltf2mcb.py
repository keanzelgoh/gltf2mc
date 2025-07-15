from gltf2mc import SimpleGLTFToMCBConverter

# Create converter
converter = SimpleGLTFToMCBConverter(
    voxel_size=0.1,     # Adjust for detail level
    max_cubes=9000      # Adjust for model complexity
)

# Convert your file
try:
    converter.convert(
        input_path="scene.gltf",
        output_path="Output/Kia Picanto.json",
        model_name="kia_picanto"
    )
    print("Success! Check your output file.")
except Exception as e:
    print(f"Error: {e}")
