Each run of perceptron fileset: trains perceptron and outputs response

bool prediction_function(float input1, float input2, std::str training_bin){

    std::vec<float> weight_vec = ...;

    float activation_variable = input1*outputweights[1] + input2*outputweights[2] + training_data[0]; //training_data[0] is bias

    return returnvalue = (activation_variable >= 0.0);
}

void training_function(std::str training_file, std::str weight_file){

    std::ofstream out_file(weight_file, std::ios::binary); //weight_file vec(bias (included), weight1, weight2);
    std::ifstream in_file(training_file, std::ios::binary);

    if (!out_file) return 1;
    if (!in_file) return 1;

    out_file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
    


    out_file.close();
    in_file.close();

}