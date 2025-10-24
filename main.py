import os
import csv

from processor import OpenCVProcessor, WandProcessor
from core import run_multi_pipeline

from monitor import wrapper_monitor

def read_csv(file_path):
    arr = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            arr.append(row)
    return arr

@wrapper_monitor()
def main(debug: bool = False):
    asset_dir = '2XL'
    file_path = '5071404_1_front.png'
    # output_dir = 'output/opencv'
    output_dir = 'output/wand'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # canvas = run_multi_pipeline(asset_dir, file_path, OpenCVProcessor, debug=debug, output_dir=output_dir)
    # canvas = run_multi_pipeline(asset_dir, file_path, WandProcessor, debug=debug , output_dir=output_dir)
    # canvas.save(os.path.join(output_dir, "final.tif"), preview=False)

    dir_path = os.path.join('.downloads')
    data = read_csv('temp.csv')
    for row in data:
        item = row['item']
        print(f'Processing item: {item}')
        file_path = os.path.join(dir_path, f'{item}_front.png')
        type = row['type']
        size = row['size']
        asset_dir = os.path.join('assets', type, size)
        
        # print(f'OpenCVProcessor processing for item: {item}')
        # output_opencv_path = os.path.join(output_dir, f'{item}_opencv.tif')
        # canvas_opencv = run_multi_pipeline(asset_dir, file_path, OpenCVProcessor)
        # canvas_opencv.save(output_opencv_path, preview=False)
        
        print(f'WandProcessor processing for item: {item}')
        output_wand_path = os.path.join(output_dir, f'{item}_wand.tif')
        canvas_wand = run_multi_pipeline(asset_dir, file_path, WandProcessor)
        print(f'Saving WandProcessor output for item: {item}')
        canvas_wand.save(output_wand_path, preview=False)
        print (f'Finished processing item: {item}')

if __name__ == "__main__":
    main(debug=True)