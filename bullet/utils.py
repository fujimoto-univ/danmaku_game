def calc_layer(input_shape, kernel_shape, stride):
    return (input_shape - kernel_shape) // stride + 1


def get_max_num_in_file(path):

    # R:114のようになっているがファイル内で右の数字の最大値を取得したい
    tmp_max = 0
    with open(path, "r") as f:
        for line in f:
            if "R:" in line:
                tmp = int(line.split("R:")[1])
                if tmp > tmp_max:
                    tmp_max = tmp
    print(tmp_max)
