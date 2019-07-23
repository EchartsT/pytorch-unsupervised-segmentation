import numpy as np
def init_sp(image, segments):
    # init graph by first method, by color distance metric between superpixels.
    row = image.shape[0]
    col = image.shape[1]

    labels = segments.reshape(image.shape[0] * image.shape[1])  # 分割后每个超像素的Sk值
    u_labels = np.unique(labels)  # 将Sk作为标签
    l_inds = []  # 每i行表示Si超像素中每个像素的编号
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])

    segmentsLabel = []
    for i in range(row):
        for j in range(col):
            l = segments[i, j]
            if l not in segmentsLabel:
                segmentsLabel.append(l)
    position = []  # 每一行记录的是属于标签i的位置信息
    ave_position = []  # i标签的坐标
    flatten_position = []  # 距离position中每个位置信息的序号

    for i in segmentsLabel:
        pixel_position = []
        flatten_pos = []
        for m in range(row):
            for n in range(col):
                if segments[m, n] == i:
                    pixel_position.append([m, n])
                    flatten_pos.append(m * col + n)

        position.append(pixel_position)
        flatten_position.append(flatten_pos)

        pixel_position = np.asarray(pixel_position)
        ave_position.append((sum(pixel_position) / len(pixel_position)).tolist())

    # generate average color value and red, green, blue color values
    average = []
    red_average = []
    green_average = []
    blue_average = []
    for i in range(len(position)):
        val = 0
        red_val = 0
        green_val = 0
        blue_val = 0
        for j in position[i]:
            [m, n] = j
            val += 0.299 * image[m, n, 0] + 0.587 * image[m, n, 1] + 0.114 * image[m, n, 2]
            red_val += image[m, n, 0]
            green_val += image[m, n, 1]
            blue_val += image[m, n, 2]
            # val += image[m, n]
        average.append(val / len(position[i]))
        red_average.append(red_val / len(position[i]))
        green_average.append(green_val / len(position[i]))
        blue_average.append(blue_val / len(position[i]))
    return ave_position, red_average, green_average, blue_average,position
