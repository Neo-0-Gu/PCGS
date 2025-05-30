from matplotlib import pyplot as plt

def draw_keypoints_on_canvas(image, keypoints, width=1, height=1, save_path="../data/pose/keypoints_image.png"):

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    
    for i, kp in enumerate(keypoints):
        if kp is None:
            continue  # 跳过空的关键点
        else:
            kp = (kp[0] * width, kp[1] * height)
            # plt.plot(x, y, 'bo')  # 红色点
            plt.text(kp[0], kp[1], f'{i}', fontsize=8, color='red')  # 用序号标注

    plt.axis('off')
    # plt.show()
    print(f"图像已保存到：{save_path}")