#include "mnist_read.h"

#include <iostream>
#include <tuple>
#include <vector>
#include <fstream>
#include <assert.h>

int reverse_int(const uint32_t& i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((uint32_t)ch1<<24)+((uint32_t)ch2<<16)+((uint32_t)ch3<<8)+ch4;
}


std::vector<std::vector<double>> my_read_mnist_imfile(
        const int& num_to_read,
        const std::string& filepath,
        const int& start_index) {

    std::vector<std::vector<double>> res;
    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        uint32_t magic_number;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        assert(magic_number == 2051);
        uint32_t num_images;
        file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
        num_images = reverse_int(num_images);
        assert(num_images >= start_index + num_to_read);
        uint32_t num_rows;
        file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
        num_rows = reverse_int(num_rows);
        uint32_t num_cols;
        file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
        num_cols = reverse_int(num_cols);
        std::vector<double> curr_image;
        if (start_index != 0) {
            //fast-forward to where we want to start
            file.seekg(start_index, std::ios::cur);
        }
        for (int i=0; i<num_to_read; i++) {
            for (int j=0; j<num_rows*num_cols; j++) {
                uint8_t currval;
                file.read(reinterpret_cast<char*>(&currval), sizeof(currval));
                double currval_normalized = (double)currval / 255.0;
                curr_image.push_back(currval_normalized);
            }
            res.push_back(curr_image);
            curr_image.clear();
        }
    }
    file.close();
    return res;
}


std::vector<int> my_read_mnist_labelfile(
        const int& num_to_read,
        const std::string& filepath,
        const int& start_index) {

    std::vector<int> res;
    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        uint32_t magic_number;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        assert(magic_number == 2049);
        uint32_t num_labels;
        file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
        num_labels = reverse_int(num_labels);
        assert(num_labels >= start_index + num_to_read);
        if (start_index != 0) {
            file.seekg(start_index, std::ios::cur);
        }
        for (int i=0; i<num_to_read; i++) {
            uint8_t currval;
            file.read(reinterpret_cast<char*>(&currval), sizeof(currval));
            res.push_back((int) currval);
        }
    }
    file.close();
    return res;
}


std::vector<std::tuple<std::vector<double>, std::vector<double>>>
massage_training_data(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) {
    assert(images.size() == labels.size());
    std::vector<std::tuple<std::vector<double>, std::vector<double>>> res;
    std::vector<double> curr_image_vec;
    std::vector<double> curr_label_vec(10);
    for (int i=0; i<images.size(); i++) {
        for (int j=0; j<images[i].size(); j++) {
            curr_image_vec.push_back(images[i][j]);
        }
        for (int j=0; j<10; j++) {
            if (labels[i] == j) {
                curr_label_vec[j] = 1.0;
            } else {
                curr_label_vec[j] = 0.0;
            }
        }
        res.push_back(
                std::tuple<std::vector<double>, std::vector<double>>(curr_image_vec, curr_label_vec));
        curr_image_vec.clear();
    }
    return res;
}


std::vector<std::tuple<std::vector<double>, int>>
massage_test_data(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) {
    assert(images.size() == labels.size());
    std::vector<std::tuple<std::vector<double>, int>> res;
    std::vector<double> curr_image_vec;
    for (int i=0; i<images.size(); i++) {
        for (int j=0; j<images[i].size(); j++) {
            curr_image_vec.push_back(images[i][j]);
        }
        res.push_back(std::tuple<std::vector<double>, int>(curr_image_vec, labels[i]));
        curr_image_vec.clear();
    }
    return res;
}


//void read_mnist(const int& num_images, const int& num_elements,
//        std::vector<std::vector<double>>& arr,
//        const std::string& filepath)
//{
//    arr.resize(num_images, std::vector<double>(num_elements));
//    std::ifstream file(filepath, std::ios::binary);
//    if (file.is_open())
//    {
//        int magic_number=0;
//        int number_of_images=0;
//        int n_rows=0;
//        int n_cols=0;
//        file.read((char*)&magic_number,sizeof(magic_number));
//        magic_number= reverse_int(magic_number);
//        file.read((char*)&number_of_images,sizeof(number_of_images));
//        number_of_images= reverse_int(number_of_images);
//        file.read((char*)&n_rows,sizeof(n_rows));
//        n_rows= reverse_int(n_rows);
//        file.read((char*)&n_cols,sizeof(n_cols));
//        n_cols= reverse_int(n_cols);
//        for(int i=0;i<number_of_images;++i)
//        {
//            for(int r=0;r<n_rows;++r)
//            {
//                for(int c=0;c<n_cols;++c)
//                {
//                    unsigned char temp=0;
//                    file.read((char*)&temp,sizeof(temp));
//                    arr[i][(n_rows*r)+c]= (double)temp;
//                }
//            }
//        }
//    }
//}
//
//
//void read_mnist_labels(const int& num_images, )

//int main()
//{
//    vector<vector<double>> ar;
//    ReadMNIST(10000,784,ar);
//
//    return 0;
//}
