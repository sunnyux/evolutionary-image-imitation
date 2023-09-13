from PIL import Image
import cv2
import numpy as np
from scipy.interpolate import griddata
# from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import KDTree


# from vangogh.fitness import draw_voronoi_matrix

NUM_VARIABLES_PER_POINT = 5
IMAGE_SHRINK_SCALE = 6

REFERENCE_IMAGE = Image.open("./img/reference_image_resized.jpg").convert('RGB')
REFERENCE_IMAGE.name = "van_gogh"

REFERENCE_IMAGE_STARRY = Image.open("./img/starry_night_resized.jpg").convert('RGB')
REFERENCE_IMAGE_STARRY.name = "starry_night"

REFERENCE_IMAGE_MONDRIAAN = Image.open("./img/mondriaan_reference_img_resized.jpg").convert('RGB')
REFERENCE_IMAGE_MONDRIAAN.name = "mondriaan"

REFERENCE_IMAGE_DONALD = Image.open("./img/donald_reference_img_resized.jpg").convert('RGB')
REFERENCE_IMAGE_DONALD.name = "donald"

REFERENCE_IMAGE_CITY = Image.open("./img/city_reference-im.jpg").convert('RGB')
REFERENCE_IMAGE_CITY.name = "city"

def pick_color(image, x, y):
    return image[y][x]


QUERY_POINTS = []

QUERY_POINTS = []


def labels_to_optimal_colored_genes(labels, feature_coords, num_points, reference_image_array, method="AVERAGE"):
    # performance can be improved by only checing the coordinates that are changed
    gene = []
    for i in range(num_points):
        pixels_of_same_color = np.where(labels==i)
        coordinates = list(zip(pixels_of_same_color[0], pixels_of_same_color[1])) # coordinates that have the lavel i

        if coordinates == []: # not the best solution but a very corner case
            # print(f"Warning: No coordinates found for label {i}")
            rng = np.random.default_rng()
            x = rng.integers(low=0, high=len(reference_image_array[0]))
            y = rng.integers(low=0, high=len(reference_image_array))
            rgb = pick_color(reference_image_array, x, y)
            gene.extend([x, y, rgb[0], rgb[1], rgb[2]])
            continue

        actual_coord = coordinates[np.random.randint(0, len(coordinates))]
        u = set(feature_coords) & set(coordinates)
        if len(u) == 1:
            actual_coord = list(u.pop())
        # else:
        #     print("Warning: There should be exactly one element in the union of feature_coords and coordinates")

        original_colors = []
        for coord in coordinates:
            original_colors.append(get_coord_color_from_image(reference_image_array, coord[0], coord[1]))
        
        if method == "AVERAGE":
            color = get_average_color(original_colors)
        elif method == "DOMINANT":
            color = get_dominant_color(original_colors)            
        else:
            raise Exception("Unknown method for picking color")
        
        gene.extend(actual_coord)
        gene.extend(color)

    return gene


def create_voronoi_site_labels(coordinates, width, height):
    grid_x, grid_y = np.mgrid[0:width, 0:height]
    labels = griddata(list(coordinates), np.arange(len(coordinates)), (grid_x, grid_y), method="nearest")
    return labels


def has_repeating_values(val):
    return len(val) != len(set(val))

def get_coordinates_only(chromosome, width, height):
    coords = set()
    l = len(chromosome)

    for m in range(0, l, NUM_VARIABLES_PER_POINT):
        coords.add((chromosome[m], chromosome[m+1]))

    if has_repeating_values(coords) or len(list(coords)) < l//NUM_VARIABLES_PER_POINT:
        # print(f"Warning: {len(coords) - len(set(coords))} repeating values in coordinates,\
        #       or not enough coordinates, {l//NUM_VARIABLES_PER_POINT} is required, only {len(list(coords))} unique coordinates were given.")
        coords = generate_unique_coordinates(l//NUM_VARIABLES_PER_POINT, width, height, coords)

    return coords

def optimize_individual_color(individual, num_points, reference_image_array, method="AVERAGE"):
    if method == "COORD":
        ret = individual.copy()
        for i in range(0, len(individual), 5):
            ret[i+2 : i+5] = get_coord_color_from_image(reference_image_array, individual[i], individual[i+1])
        return ret
    else:
        width = len(reference_image_array[0]) # 50
        height = len(reference_image_array) # 39
        coords = get_coordinates_only(individual, width, height)
        labels = create_voronoi_site_labels(coords, width, height)
        return labels_to_optimal_colored_genes(labels, coords, num_points, reference_image_array, method=method)


def generate_unique_coordinates(num_points, width, height, coords=set()):
    feature_coords = coords
    while len(feature_coords) < num_points:
        rng = np.random.default_rng()
        xs = rng.integers(low=0, high=width, size=num_points - len(feature_coords))
        ys = rng.integers(low=0, high=height, size=num_points - len(feature_coords))
        cs = set(zip(xs, ys))        
        feature_coords = feature_coords | cs
    return feature_coords

    
def initialize_one_with_best_voronoi_color(feature_intervals, reference_image_array, method="AVERAGE"):
    width = len(reference_image_array[0]) # 50
    height = len(reference_image_array) # 39
    genotype_length = len(feature_intervals)
    num_points = genotype_length // 5

    feature_coords = generate_unique_coordinates(num_points, width, height)

    labels = create_voronoi_site_labels(feature_coords, width, height)

    gene = labels_to_optimal_colored_genes(labels, feature_coords, num_points, reference_image_array, method)

    return gene


def get_voronoi_polygon_sites(genotype, width, height):
    np.set_printoptions(threshold=np.inf)

    extracted_elements = []
    for i in range(0, len(genotype), 5):
        extracted_elements.append(tuple(genotype[i:i+2]))

    grid_x, grid_y = np.mgrid[0:width, 0:height]
    labels = griddata(extracted_elements, np.arange(len(extracted_elements)), (grid_x, grid_y), method="nearest")

    return labels

def get_coord_color_from_image(reference_image_array, x, y):
    return reference_image_array[y][x]

def get_average_color(colors):
    average = np.mean(colors, axis=0, dtype=int)
    return average

def get_dominant_color(clrs):
    if len(clrs) < 3: # otherwise errors
        idx = np.random.randint(0, len(clrs))
        # print(f"clrs: {clrs}, idx: {idx}")
        return clrs[idx]
    
    colors = np.array(clrs, dtype=np.float32)
    
    n_colors = 3 # same as average when n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(colors, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    # print(f"dominant: {dominant}")
    return dominant

# def pick_average_voronoi_color(genotype_coords, reference_image_array, polygon_coord, scale=IMAGE_SHRINK_SCALE):
#     pixel_colors = draw_voronoi_matrix_and_get_polygon_colors(genotype_coords, reference_image_array, polygon_coord, scale)
#     average = np.mean(pixel_colors, axis=0)
#     # print(f"average: {average}")
#     return average

# def pick_dominant_voronoi_color(genotype_coords, reference_image_array, polygon_coord, scale=IMAGE_SHRINK_SCALE):
#     # does not perform as well compare to average
#     pixel_colors = draw_voronoi_matrix_and_get_polygon_colors(genotype_coords, reference_image_array, polygon_coord, scale)

#     if len(pixel_colors) == 1:
#         return pixel_colors[0]
    
#     n_colors = 3 # same as average when n_colors = 1
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#     # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
#     flags = cv2.KMEANS_RANDOM_CENTERS

#     _, labels, palette = cv2.kmeans(pixel_colors, n_colors, None, criteria, 10, flags)
#     _, counts = np.unique(labels, return_counts=True)

#     dominant = palette[np.argmax(counts)]

#     # print(f"dominant: {dominant}")
#     return dominant




# def draw_voronoi_matrix_and_get_polygon_colors(genotype, reference_image_array, polygon_coord, scale=1):
#     # the performance of this is really bad
#     # np.set_printoptions(threshold=np.inf)
    
#     img_width = len(reference_image_array[0]) # 50
#     img_height = len(reference_image_array) # 39

#     scaled_img_width = int(img_width * scale)
#     scaled_img_height = int(img_height * scale)
#     num_points = int(len(genotype) / NUM_VARIABLES_PER_POINT)
#     coords = []
#     colors = []
#     polygon_coord_color = (0,0,0)

#     for r in range(num_points):
#         p = r * NUM_VARIABLES_PER_POINT
#         x, y, r, g, b = genotype[p:p + NUM_VARIABLES_PER_POINT]
#         coords.append((x * scale, y * scale))
#         colors.append((r, g, b))
#         # print(f"x: {x}, y: {y}, polygon_coord: {polygon_coord}")
#         if (x, y) == polygon_coord:
#             polygon_coord_color = (r, g, b)

#     voronoi_kdtree = KDTree(coords)
#     if scale == 1:
#         query_points = QUERY_POINTS
#     else:
#         query_points = [(x, y) for x in range(scaled_img_width) for y in range(scaled_img_height)]

#     _, query_point_regions = voronoi_kdtree.query(query_points)

#     c_x, c_y = polygon_coord
#     polygon_orig_colors = []
#     i = 0
#     for x in range(scaled_img_width):
#         for y in range(scaled_img_height):
#             if polygon_coord_color == colors[query_point_regions[i]]:
#                 polygon_orig_colors.append(reference_image_array[int(y/scale)][int(x/scale)])
#             i += 1

#     if polygon_orig_colors == []:
#         polygon_orig_colors.append(reference_image_array[c_y][c_x])

#     pixel_colors = np.float32(np.array(polygon_orig_colors))
#     return pixel_colors


# def get_colors_from_image(image, points):
#     pt = np.transpose(points)
#     x_coords = pt[0]
#     y_coords = pt[1]
#     return image[x_coords, y_coords]