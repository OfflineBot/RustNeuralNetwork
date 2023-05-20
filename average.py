import os

path = "mnist_images"

for i in range(10):
    x = f"{path}/{i}"
    y = os.listdir(x)
    folder_path = x
    for num in range(1, len(y)):
        current_name = os.path.join(folder_path, str(num))
        new_name = os.path.join(folder_path, str(num))

        # Check if the file with the current number exists
        if not os.path.exists(current_name):
            # Find the previous number that exists
            prev_num = num - 1
            while prev_num > 0 and not os.path.exists(os.path.join(folder_path, str(prev_num))):
                prev_num -= 1

            # Rename the file to the previous number
            if prev_num > 0:
                new_name = os.path.join(folder_path, str(prev_num))

    # Rename the file
    os.rename(current_name, new_name)
    print(f"Renamed {current_name} to {new_name}")
    print(len(y))
