import math

from PIL import Image
from shapely.geometry import LineString, MultiLineString, Point, mapping
from shapely.ops import unary_union

class Node_surface:

    def __init__(self, points):
        self.net = points

    def set_surface_nodes(self):

        # for x, y, z in points:

        for star in self.filtered_structure:
            updated_connections = []
            for direction, line in star.connections:
                new_coords = []
                for x, y, *rest in line.coords:
                    # find all stars in the XY neighborhood of this endpoint
                    neighbors = [
                        s for sx, sy, s in star_positions
                        if abs(sx - x) <= grid_size / 2
                        and abs(sy - y) <= grid_size / 2
                    ]
                    if neighbors:
                        top_star = min(neighbors, key=lambda s: s.point.z)
                        new_coords.append((x, y, top_star.point.z))
                    else:
                        # no neighbor found, keep original z
                        if rest:
                            new_coords.append((x, y, rest[0]))
                        else:
                            new_coords.append((x, y, 0.0))
                updated_connections.append((direction, LineString(new_coords)))
            star.connections = updated_connections

        return
