from cmath import inf
import numpy as np
from PIL import Image

if __name__ == '__main__':
    image = Image.open("train/train.jpg")
    arr_img = np.array(image)
    image_width = image.width
    image_height = image.height
    channels = 3 

    alpha = 0.0007 # Learning rate
    max_error = 30

    block_width , block_height = 4, 4 # Block size
    blocks_count =  image_width // block_width * image_height // block_height
    roll_size = block_width * block_height * channels 

    code_size = int(roll_size // 2)

    compression_ratio = (roll_size * blocks_count) / ((roll_size + blocks_count) * code_size + 2)

    print("Blocks size: ", block_width, block_height)
    print("Compression_ration: ", compression_ratio)
    print("Name: ", image.filename)
    print("Learning rate: ", alpha)
    print("Error: ", max_error)


    etalons = np.empty((image_width * image_height * channels // roll_size, 1, roll_size))
    counter = 0
    for row in range(0, image_height, block_height):
        for col in range(0, image_width, block_width):
            block = np.empty(roll_size)

            for block_row in range(block_height):
                for block_col in range(block_width):
                    for channel in range(channels):
                        block[channel + channels * (block_row + block_col * block_height)] = arr_img[block_row + row,
                                                                                                     block_col + col][channel]

            block = block / 255 * 2 - 1
            etalons[counter] = block
            counter += 1
            
  
    output = np.empty((blocks_count, 1, roll_size))
    delta = np.empty((blocks_count, 1, roll_size))

    np.random.seed(0)

    answer = 0
    while answer != 'Y' and answer != 'N':
        answer = str(input("Do you want to load weights?[Y/N]:"))

    if answer.lower() == 'y':
        weight1 = np.load("weight1.npy")
        weight2 = np.load("weight2.npy")

    else:
        weight1 = np.random.uniform(size=(roll_size, code_size), low=-1, high=1)
        weight1.dtype = np.float64

        weight2 = np.random.uniform(size=(code_size, roll_size), low=-1, high=1)
        weight2.dtype = np.float64

        counter = 0
        MSE_error = inf
        while max_error < MSE_error:
            MSE_error = 0

            for k in range(blocks_count):
                y = etalons[k] @ weight1

                output[k] = y @ weight2
                delta[k] = output[k] - etalons[k]
            
                tmp_weight2 = weight2 - alpha * np.transpose(y) @ delta[k]
                tmp_weight1 = weight1 - alpha * np.transpose(etalons[k]) @ delta[k] @ np.transpose(weight2)

                weight1 = tmp_weight1
                weight2 = tmp_weight2

            # Evaluate error
            for k in range(blocks_count):
                y = etalons[k] @ weight1
                output[k] = y @ weight2
                delta[k] = output[k] - etalons[k]

                for i in range(roll_size):
                    MSE_error += delta[k][0][i] * delta[k][0][i]

            counter += 1
            print(f"Epoch #{counter}, error = {MSE_error}")

        print(f"Epoch #{counter}, error = {MSE_error}")

    #if answer.lower() == 'n':
    #    np.save("weight1.npy", weight1)
    #    np.save("weight2.npy", weight2)

    if (answer.lower() != 'y'):
        answer = 0
        while answer != 'Y' and answer != 'N':
            answer = str(input("Do you want to save weights?[Y/N]:"))

        if answer.lower() == 'y':
            np.save("weight1.npy", weight1)
            np.save("weight2.npy", weight2)

        compressed_etalons = []

        for k in range(blocks_count):
            y = etalons[k] @ weight1
            compressed_etalons.append(y)
            output[k] = y @ weight2

    image_restored = np.empty((image_height, image_width, channels))
   
    for row in range(0, image_height, block_height):
        for col in range(0, image_width, block_width):
            output_value = output[int((row / block_height) * image_height / block_height + (col / block_width))][0]

            for j in range(block_height):
                for k in range(block_width):
                    for i in range(channels):
                        image_restored[j + row, k + col, i] = 255 * (output_value[i + channels * (j + k * block_height)] + 1) / 2

    result = Image.fromarray(image_restored.astype(np.uint8))
    result.save("result/output.jpg")

    compression_ratio = (roll_size * blocks_count) / ((roll_size + blocks_count) * code_size + 2)

    print("Name: ", image.filename)
    print("Learning rate: ", alpha)
    print("Error: ", max_error)
    print("Compression ratio: ", compression_ratio)
