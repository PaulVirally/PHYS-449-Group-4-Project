'''
PHYS 449 -- Fall 2022
Workshop 1: 
    - Python classes
    - Proper source code structure
    - Reading / writing legibly with .json files
    - Command line 
'''

import numpy as np # loading in b values
import json, argparse, sys
import matplotlib.pyplot as plt

# this points us to the meat and potatos of the code
sys.path.append('src') 
from rectangle import Rectangle

def plot_results(area_info, plot_file_name):
    num_rectangles = len(area_info)

    fig, (ax_area, ax_perimeter) = plt.subplots(1, 2)
    for i in range(num_rectangles):
        rectangle_i = area_info["Rectangle{}".format(i+1)]
        ax_area.scatter(rectangle_i["b"], rectangle_i["area"], color='blue')
        ax_perimeter.scatter(rectangle_i["b"], rectangle_i["perimeter"], color='red')

    ax_area.set_title("Area vs. b")
    ax_perimeter.set_title("Perimeter vs. b")

    fig.savefig(plot_file_name, dpi=500)

def calculate_all_info(a, b_vals):
    # b, perimeter, area
    shape_info = {}
    for i,b in enumerate(b_vals):
        rect = Rectangle(a, b)
        area = rect.area()
        perimeter = rect.perimeter()

        shape_info['Rectangle{}'.format(i+1)] = {}
        shape_info['Rectangle{}'.format(i+1)]["b"] = b
        shape_info['Rectangle{}'.format(i+1)]["area"] = area
        shape_info['Rectangle{}'.format(i+1)]["perimeter"] = perimeter

    return shape_info

if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='Tutorial 1')

    parser.add_argument('-a', default=2.0, type=float, help="The first dimension of the rectangle.")
    parser.add_argument('--b-vals-path', help='A file/path containing b-values for the rectangle.')
    parser.add_argument('--res-path',
                        help='Path to results')
    
    args = parser.parse_args()

    b_vals = np.loadtxt(args.b_vals_path)
    a = args.a

    shape_info = calculate_all_info(a, b_vals)
    
    results_file = args.res_path + "a={}_rectangles.json".format(a)
    with open(results_file, 'w') as f:
        json.dump(shape_info, f, indent=4)
    
    plot_file_name = args.res_path + "a={}_b_varies.pdf".format(a)
    plot_results(shape_info, plot_file_name)