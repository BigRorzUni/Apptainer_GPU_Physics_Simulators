import xml.etree.ElementTree as ET
import numpy as np
import sys

def generate_random_spheres_xml(n=10):
    mjcf = ET.Element("mujoco", model="random_spheres")

    filename = f"../xml/random_spheres_{n}.xml"

    # Option settings
    ET.SubElement(mjcf, "option", timestep="0.01", gravity="0 0 -9.81")

    # Worldbody
    worldbody = ET.SubElement(mjcf, "worldbody")

    # Add a static ground plane
    ET.SubElement(worldbody, "geom", name="floor", type="plane", size="50 50 0.1", pos="0 0 0", rgba="0.8 0.8 0.8 1")

    # Add n spheres
    for i in range(n):
        x, y = np.random.uniform(-2, 2), np.random.uniform(-2, 2)
        z = 1.0  # height above ground
        body = ET.SubElement(worldbody, "body", name=f"sphere_{i}", pos=f"{x:.3f} {y:.3f} {z}")
        ET.SubElement(body, "geom", type="sphere", size="0.1", rgba="0.2 0.4 0.6 1")
        ET.SubElement(body, "freejoint")

    # Save to file
    tree = ET.ElementTree(mjcf)
    ET.indent(tree, space="  ", level=0)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
    print(f"Saved MJCF with {n} spheres to '{filename}'")

# Example usage
if __name__ == "__main__":
    n = 20
    if len(sys.argv) >= 2:
        n = int(sys.argv[1])

    generate_random_spheres_xml(n)