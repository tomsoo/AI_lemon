import csv

def split_valid_test_csv():
    with open('dataset/eval_images.csv', 'r') as f:
        reader = csv.reader(f)
        csv_data = [low for low in reader]
    csv_label = csv_data.pop(0)
    csv_data_num = len(csv_data)
    csv_data_valid = [csv_label] + csv_data[:int(csv_data_num/2)]
    csv_data_infer = [csv_label] + csv_data[int(csv_data_num/2):]

    list_to_csv('dataset/valid_images.csv',csv_data_valid,2)
    list_to_csv('dataset/infer_images.csv',csv_data_infer,2)

def list_to_csv(csv_file_name,array,dim):
    with open(csv_file_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        if dim == 1:
            writer.writerow(array)     # list（1次元配列）の場合
        elif dim == 2:
            writer.writerows(array) # 2次元配列も書き込める
        else:
            print('error: You must chose dim 1 or 2')
            exit(1)

if __name__=='__main__':
    split_valid_test_csv()
